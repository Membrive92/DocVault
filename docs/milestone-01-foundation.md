# Milestone 1: Foundation

> **Status:** ✅ COMPLETED
> **Duration:** Initial setup
> **Complexity:** Low

---

## Objective

Establish the base project structure and flexible configuration system that will support all future milestones.

## Goals

1. Create modular project structure
2. Implement type-safe configuration with Pydantic
3. Set up environment variables management
4. Create verification script
5. Document project setup

---

## Deliverables

### 1. Project Structure

```
DocVault/
├── config/              # Centralized configuration
│   ├── __init__.py
│   └── settings.py      # Pydantic Settings
├── src/                 # Source code
│   └── __init__.py
├── tests/               # Test files (pytest)
├── scripts/             # Verification scripts
├── data/
│   ├── documents/       # Documents to ingest
│   └── qdrant_storage/  # Vector DB (future)
├── .env.example         # Environment variables template
├── .gitignore
├── requirements.txt
├── test_setup.py        # Verification script
├── README.md
└── AGENTS.md           # Guide for AI agents
```

**Key Decisions:**
- **Modular structure:** Each milestone gets its own folder in `src/`
- **Separate config:** Configuration isolated in `config/` module
- **Scripts folder:** Interactive verification scripts separate from tests
- **Data folder:** Keep data separate from code

---

### 2. Configuration System

**File:** `config/settings.py`

**Implementation:**
```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # General
    project_name: str = Field(default="docvault")
    environment: Literal["development", "production", "testing"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    documents_dir: Path = Field(default=Path("data/documents"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def get_full_path(self, relative_path: Path) -> Path:
        """Convert relative path to absolute based on project_root."""
        if relative_path.is_absolute():
            return relative_path
        return self.project_root / relative_path

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.get_full_path(self.data_dir),
            self.get_full_path(self.documents_dir)
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global instance
settings = Settings()
```

**Why Pydantic Settings?**
1. **Type safety:** Automatic validation and type conversion
2. **Environment variables:** Automatic loading from `.env`
3. **Default values:** Sensible defaults for development
4. **IDE support:** Autocomplete and type hints
5. **Extensible:** Easy to add new settings in future milestones

---

### 3. Environment Variables

**File:** `.env.example`

```env
# General
PROJECT_NAME=docvault
ENVIRONMENT=development
LOG_LEVEL=INFO

# Paths
DATA_DIR=data
DOCUMENTS_DIR=data/documents

# Future milestones will add:
# LLM_PROVIDER=ollama_local
# LLM_MODEL=llama3.2:3b
# OPENAI_API_KEY=sk-...
```

**Usage:**
```bash
# Copy template
cp .env.example .env

# Edit as needed
nano .env
```

---

### 4. Verification Script

**File:** `test_setup.py`

```python
"""Verify that Milestone 1 is correctly set up."""

from config.settings import settings

def main():
    print("\n" + "="*50)
    print("  DocVault - Setup Verification")
    print("="*50)

    # Test 1: Configuration loads
    try:
        assert settings.project_name == "docvault"
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

    # Test 2: Directories exist
    try:
        settings.ensure_directories()
        print("✅ Directories created successfully")
    except Exception as e:
        print(f"❌ Directory error: {e}")
        return False

    # Test 3: Display config
    settings.display_config()

    print("\n🎉 Everything is configured correctly!")
    print("📝 Next step: Milestone 2 - Embeddings\n")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

**Verification:**
```bash
python test_setup.py

# Expected output:
# ==================================================
#   DocVault - Setup Verification
# ==================================================
# ✅ Configuration loaded successfully
# ✅ Directories created successfully
#
# ==================================================
#   DOCVAULT - Configuration
# ==================================================
# Environment:      development
# Log Level:        INFO
# Project Root:     /path/to/DocVault
# Data Dir:         /path/to/DocVault/data
# Documents Dir:    /path/to/DocVault/data/documents
# ==================================================
#
# 🎉 Everything is configured correctly!
# 📝 Next step: Milestone 2 - Embeddings
```

---

## Implementation Steps

### Step 1: Create Basic Structure
```bash
mkdir -p config src tests scripts data/documents
touch config/__init__.py config/settings.py
touch src/__init__.py
touch test_setup.py
```

### Step 2: Install Dependencies
```bash
pip install pydantic pydantic-settings python-dotenv PyYAML
pip freeze > requirements.txt
```

### Step 3: Implement Configuration
- Create `config/settings.py` with Pydantic Settings
- Create `.env.example` template
- Test configuration loading

### Step 4: Create Verification Script
- Implement `test_setup.py`
- Test directory creation
- Test configuration display

### Step 5: Documentation
- Create comprehensive README.md
- Create AGENTS.md for AI agents
- Document all setup steps

---

## Key Learnings

### 1. Pydantic Settings is Powerful

**Before (manual config):**
```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "docvault")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# ... lots of boilerplate
```

**After (Pydantic Settings):**
```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "docvault"
    environment: str = "development"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()  # That's it!
```

**Benefits:**
- Automatic type conversion
- Validation out of the box
- Clear defaults
- Self-documenting

---

### 2. Configuration Hierarchy

**Priority order:**
1. System environment variables (highest)
2. `.env` file
3. Default values in `Settings` class (lowest)

**Example:**
```bash
# In .env
PROJECT_NAME=my_custom_name

# In code
settings.project_name  # "my_custom_name" (from .env)
```

```bash
# System environment variable
export PROJECT_NAME=override_name

# In code
settings.project_name  # "override_name" (system env wins)
```

---

### 3. Path Handling

**Always use `pathlib.Path`:**
```python
# ❌ BAD
path = "data/documents/" + filename
path = os.path.join("data", "documents", filename)

# ✅ GOOD
path = settings.documents_dir / filename
path = Path(settings.documents_dir) / filename
```

**Why?**
- Cross-platform compatibility (Windows vs Unix)
- Clean API
- Type safety
- Modern Python standard

---

## Testing

### Manual Testing
```bash
# 1. Verify installation
python test_setup.py

# 2. Check configuration
python -c "from config.settings import settings; settings.display_config()"

# 3. Test directory creation
python -c "from config.settings import settings; settings.ensure_directories()"
ls -la data/
```

### Automated Testing (Future)
```python
# tests/test_config.py
from config.settings import Settings

def test_default_values():
    settings = Settings()
    assert settings.project_name == "docvault"
    assert settings.environment == "development"

def test_env_file_loading():
    settings = Settings(_env_file=".env.test")
    # test loaded values
```

---

## Next Milestone

**M2: Local Embeddings**

With the foundation in place, we can now:
1. Add new modules to `src/` (e.g., `src/embeddings/`)
2. Add new settings to `config/settings.py`
3. Add new environment variables to `.env`
4. Follow the same patterns for consistency

**Preview of M2 changes:**
```python
# config/settings.py (M2 additions)
class Settings(BaseSettings):
    # ... existing fields ...

    # M2: Embedding configuration
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_dimension: int = Field(default=384)
```

---

## Conclusion

Milestone 1 established:
- ✅ Clean, modular project structure
- ✅ Type-safe configuration system
- ✅ Environment variables management
- ✅ Verification mechanism
- ✅ Comprehensive documentation

This foundation makes all future milestones easier to implement and maintain.
