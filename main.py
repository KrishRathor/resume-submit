import os
from typing import Annotated, Optional
from pathlib import Path
import mimetypes

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair

import markdownify
import readabilipy

from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent

from pydantic import BaseModel as PydanticBaseModel, AnyUrl, Field  # using pydantic directly


# ==========
# CONFIG
# ==========
# Prefer environment variables so you don’t accidentally commit secrets
TOKEN = os.getenv("PUCH_TOKEN", "<replace_with_application_key>")  # from /apply <reply-url>
MY_NUMBER = os.getenv("PUCH_PHONE", "919876543210")  # e.g. 91 + number (no '+')
RESUME_PATH = Path(os.getenv("RESUME_PATH", "./resume.md"))  # can be .md/.pdf/.docx/.html/.txt

if TOKEN.startswith("<replace"):
    raise RuntimeError("Set PUCH_TOKEN env var to your application key from /apply")

if not (MY_NUMBER.isdigit() and 8 <= len(MY_NUMBER) <= 15):
    raise RuntimeError("Set PUCH_PHONE env var to digits only (countrycode + number), e.g. 9198xxxxxx")


# ==========
# AUTH
# ==========
class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    Minimal bearer provider that validates a single known token.
    Matches the gist’s suggested approach.  :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=[],
                expires_at=None,
            )
        return None


# ==========
# FETCH helper
# ==========
class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(cls, url: str, user_agent: str, force_raw: bool = False) -> tuple[str, str]:
        """
        Return (content_for_llm, prefix_info)
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                resp = await client.get(url, follow_redirects=True, headers={"User-Agent": user_agent}, timeout=30)
            except HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if resp.status_code >= 400:
                raise McpError(
                    ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status {resp.status_code}")
                )

            text = resp.text
            ctype = resp.headers.get("content-type", "")

            is_html = ("<html" in text[:200].lower()) or ("text/html" in ctype) or (ctype == "")
            if is_html and not force_raw:
                return cls.extract_content_from_html(text), ""
            return (text, f"Content type {ctype} cannot be simplified to markdown, returning raw content:\n")

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        result = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not result.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        md = markdownify.markdownify(result["content"], heading_style=markdownify.ATX)
        return md


# ==========
# MCP init
# ==========
mcp = FastMCP("My MCP Server", auth=SimpleBearerAuthProvider(TOKEN))


# ==========
# Tool descriptions
# ==========
class RichToolDescription(PydanticBaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None


ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="When Puch (or anyone) asks for your resume; must return raw markdown, no extra formatting.",
    side_effects=None,
)

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return simplified content (markdown) or raw HTML when requested.",
    use_when="Use when a URL's contents are needed in-chat.",
    side_effects="Returns truncated content if too long; caller can paginate via start_index.",
)


# ==========
# Resume conversion helpers
# ==========
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed reading text file: {e}"))


def _read_pdf_as_markdown(p: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(p)) or ""
        if not text.strip():
            return "<error>PDF contained no extractable text</error>"
        # Minimal heuristics to markdown-ize paragraphs
        lines = [ln.strip() for ln in text.splitlines()]
        blocks = []
        para = []
        for ln in lines:
            if ln:
                para.append(ln)
            else:
                if para:
                    blocks.append(" ".join(para))
                    para = []
        if para:
            blocks.append(" ".join(para))
        return "\n\n".join(blocks)
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed reading PDF: {e}"))


def _read_docx_as_markdown(p: Path) -> str:
    try:
        import docx  # python-docx
        doc = docx.Document(str(p))
        paras = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paras)
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed reading DOCX: {e}"))


def _read_html_as_markdown(p: Path) -> str:
    try:
        html = p.read_text(encoding="utf-8")
        return Fetch.extract_content_from_html(html)
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed reading HTML: {e}"))


def _resume_to_markdown(p: Path) -> str:
    if not p.exists():
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Resume file not found: {p}"))

    # Guess by suffix first; fallback to mimetype
    ext = p.suffix.lower()
    if ext == ".md":
        return _read_text_file(p)
    if ext == ".txt":
        return _read_text_file(p)
    if ext == ".pdf":
        return _read_pdf_as_markdown(p)
    if ext == ".docx":
        return _read_docx_as_markdown(p)
    if ext in (".html", ".htm"):
        return _read_html_as_markdown(p)

    # Fallback by mime
    mime, _ = mimetypes.guess_type(p.name)
    if mime == "text/markdown":
        return _read_text_file(p)
    if mime and mime.startswith("text/"):
        return _read_text_file(p)

    # Last resort: read as utf-8 text
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unsupported resume format for file: {p.name}")
        )


# ==========
# Tools
# ==========
@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text (no extra wrapping).
    Required by the gist’s instructions.  :contentReference[oaicite:2]{index=2}
    """
    md = _resume_to_markdown(RESUME_PATH)

    # Ensure it's plain markdown string with no front-matter or extra enclosing objects
    # (Puch expects a raw string, not JSON-wrapped content)
    return md


@mcp.tool
async def validate() -> str:
    """
    Must exist for Puch validation (returns your phone number as digits only).
    """
    return MY_NUMBER


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[int, Field(default=5000, gt=0, lt=1_000_000,
                                     description="Maximum characters to return.")] = 5000,
    start_index: Annotated[int, Field(default=0, ge=0,
                                      description="Start offset to continue fetching paginated content.")] = 0,
    raw: Annotated[bool, Field(default=False, description="Return raw HTML content if true.")] = False,
) -> list[TextContent]:
    url_str = str(url).strip()
    if not url_str:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_len = len(content)

    if start_index >= original_len:
        content = "<error>No more content available.</error>"
    else:
        chunk = content[start_index:start_index + max_length]
        if not chunk:
            content = "<error>No more content available.</error>"
        else:
            content = chunk
            actual_len = len(chunk)
            remaining = original_len - (start_index + actual_len)
            if actual_len == max_length and remaining > 0:
                next_start = start_index + actual_len
                content += f"\n\n<error>Content truncated. Call fetch with start_index={next_start} to continue.</error>"

    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]


# ==========
# Entrypoint (HTTP transport)
# ==========
async def main():
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8085)  # exposes /mcp


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

