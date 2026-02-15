"""
Setup verification script for Lacuna backend.
Checks all dependencies and services are properly configured.
"""
import asyncio
import sys
import os
from typing import List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_status(message: str, status: bool):
    """Print colored status message."""
    symbol = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    print(f"{symbol} {message}")


async def check_python_version() -> bool:
    """Check Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_status(f"Python version: {version.major}.{version.minor}.{version.micro}", True)
        return True
    else:
        print_status(f"Python version {version.major}.{version.minor} (requires 3.11+)", False)
        return False


async def check_dependencies() -> bool:
    """Check if required packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "asyncpg",
        "httpx",
        "fitz",
        "docx",
        "pytesseract",
        "hdbscan",
        "pgvector",
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"Package '{package}' installed", True)
        except ImportError:
            print_status(f"Package '{package}' missing", False)
            all_installed = False

    return all_installed


async def check_env_file() -> bool:
    """Check if .env file exists."""
    if os.path.exists(".env"):
        print_status(".env file exists", True)
        return True
    else:
        print_status(".env file missing (copy from .env.example)", False)
        return False


async def check_upload_dir() -> bool:
    """Check if upload directory exists."""
    if os.path.exists("./uploads"):
        print_status("Upload directory exists", True)
        return True
    else:
        print_status("Upload directory missing (will be created on startup)", False)
        return False


async def check_ollama() -> bool:
    """Check if Ollama is running and has required models."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")

            if response.status_code == 200:
                print_status("Ollama service is running", True)

                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                # Check for required models
                has_embed = any("nomic-embed-text" in name for name in model_names)
                has_llm = any("qwen2.5:3b" in name or "qwen2.5" in name for name in model_names)

                print_status(f"Embedding model (nomic-embed-text): {'Found' if has_embed else 'Missing'}", has_embed)
                print_status(f"LLM model (qwen2.5:3b): {'Found' if has_llm else 'Missing'}", has_llm)

                return has_embed and has_llm
            else:
                print_status(f"Ollama service error (status {response.status_code})", False)
                return False

    except Exception as e:
        print_status(f"Ollama connection failed: {str(e)}", False)
        print(f"  {YELLOW}Make sure Ollama is installed and running{RESET}")
        print(f"  {YELLOW}Install from: https://ollama.ai/{RESET}")
        return False


async def check_postgres() -> bool:
    """Check if PostgreSQL is running."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            "postgresql://lacuna_user:lacuna_password@localhost:5432/lacuna_db",
            timeout=5
        )

        # Check pgvector extension
        result = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
        )

        await conn.close()

        print_status("PostgreSQL connection successful", True)
        print_status(f"pgvector extension: {'Installed' if result > 0 else 'Missing'}", result > 0)

        return result > 0

    except Exception as e:
        print_status(f"PostgreSQL connection failed: {str(e)}", False)
        print(f"  {YELLOW}Run: docker-compose up -d{RESET}")
        return False


async def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Lacuna Backend - Setup Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    checks: List[Tuple[str, callable]] = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Upload Directory", check_upload_dir),
        ("PostgreSQL + pgvector", check_postgres),
        ("Ollama + Models", check_ollama),
    ]

    results = []

    for check_name, check_func in checks:
        print(f"\n{BLUE}Checking {check_name}...{RESET}")
        try:
            result = await check_func()
            results.append(result)
        except Exception as e:
            print_status(f"Error during check: {str(e)}", False)
            results.append(False)

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"{GREEN}✓ All checks passed! ({passed}/{total}){RESET}")
        print(f"\n{GREEN}You're ready to run the backend:{RESET}")
        print(f"  python run.py")
        print(f"  or")
        print(f"  uvicorn app.main:app --reload")
    else:
        print(f"{RED}✗ Some checks failed ({passed}/{total} passed){RESET}")
        print(f"\n{YELLOW}Please fix the issues above before running the backend.{RESET}")
        sys.exit(1)

    print(f"{BLUE}{'='*60}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
