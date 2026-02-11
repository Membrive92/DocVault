"""
Setup Verification Script - Milestone 1

This script verifies that everything is correctly configured.
Run: python test_setup.py
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_imports():
    """Verify that required libraries are installed."""
    print("\n🔍 Verifying imports...")

    required_packages = {
        "pydantic": "Data validation",
        "pydantic_settings": "Configuration management",
        "dotenv": "Environment variables",
        "yaml": "YAML reading"
    }

    failed = []

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package:20} - {description}")
        except ImportError:
            print(f"  ✗ {package:20} - {description} (NOT INSTALLED)")
            failed.append(package)

    if failed:
        print("\n❌ Missing packages. Install with:")
        print("   pip install -r requirements.txt")
        return False

    print("✅ All packages are installed\n")
    return True


def test_config():
    """Verify that configuration system works."""
    print("🔍 Verifying configuration...")

    try:
        from config.settings import settings

        print(f"  ✓ Configuration loaded successfully")
        print(f"  ✓ Project: {settings.project_name}")
        print(f"  ✓ Environment: {settings.environment}")
        print(f"  ✓ Project root: {settings.project_root}")

        # Display full configuration
        settings.display_config()

        print("✅ Configuration system working\n")
        return True

    except Exception as e:
        print(f"  ✗ Error loading configuration: {e}")
        return False


def test_directories():
    """Verify and create necessary directories."""
    print("🔍 Verifying directories...")

    try:
        from config.settings import settings

        # Create directories
        settings.ensure_directories()

        # Verify they exist
        dirs_to_check = [
            settings.get_full_path(settings.data_dir),
            settings.get_full_path(settings.documents_dir)
        ]

        all_exist = True
        for directory in dirs_to_check:
            if directory.exists():
                print(f"  ✓ {directory}")
            else:
                print(f"  ✗ {directory} (does not exist)")
                all_exist = False

        if all_exist:
            print("✅ All directories are ready\n")
            return True
        else:
            print("❌ Some directories were not created\n")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return False


def test_env_file():
    """Verify if .env file exists"""
    print("🔍 Verifying .env file...")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print(f"  ✓ .env file found")
        print("✅ Environment configuration ready\n")
        return True
    elif env_example.exists():
        print(f"  ⚠️  .env file not found")
        print(f"  ℹ️  You can create one by copying .env.example:")
        print(f"     cp .env.example .env")
        print("⚠️  Continuing with default values\n")
        return True
    else:
        print(f"  ✗ Neither .env nor .env.example found")
        return False


def main():
    """Run all verifications."""
    print("\n" + "=" * 60)
    print("  🚀 DocVault - Setup Verification (Milestone 1)")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        (".env File", test_env_file),
        ("Configuration", test_config),
        ("Directories", test_directories)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}\n")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("  📊 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 Everything is configured correctly!")
        print("📝 Next step: Milestone 2 - Embeddings")
        return 0
    else:
        print("\n⚠️  There are issues to resolve before continuing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
