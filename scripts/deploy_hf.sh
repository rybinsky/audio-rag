#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SPACE_NAME="audio-rag-demo"
OUTPUT_DIR="./hf_deploy"
DRY_RUN=false
SKIP_CONFIRM=false

# Help message
usage() {
    cat << EOF
🎙️  Audio RAG - Hugging Face Spaces Deployment

Usage: $0 [OPTIONS]

Options:
    -u, --username        Hugging Face username (required, or set HF_USERNAME)
    -s, --space-name      Name of the Space (default: audio-rag-demo)
    -t, --token           Hugging Face API token (required, or set HF_TOKEN)
    -o, --output-dir      Output directory (default: ./hf_deploy)
    -n, --dry-run         Prepare files but don't push
    -y, --yes             Skip confirmation
    -h, --help            Show this help message

Examples:
    # With command line arguments
    $0 --username johndoe --token hf_xxx --space-name audio-rag-demo

    # With environment variables
    export HF_USERNAME=johndoe
    export HF_TOKEN=hf_xxx
    $0

    # Dry run
    $0 --username johndoe --dry-run

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            HF_USERNAME="$2"
            shift 2
            ;;
        -s|--space-name)
            SPACE_NAME="$2"
            shift 2
            ;;
        -t|--token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check dependencies
command -v git >/dev/null 2>&1 || {
    echo -e "${RED}Error: git is required but not installed.${NC}"
    exit 1
}

# Validate arguments
if [ "$DRY_RUN" = false ]; then
    if [ -z "$HF_USERNAME" ]; then
        echo -e "${RED}Error: Username required (use --username or set HF_USERNAME)${NC}"
        exit 1
    fi
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: Token required (use --token or set HF_TOKEN)${NC}"
        exit 1
    fi
fi

# Print header
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}🎙️  Audio RAG - Hugging Face Spaces Deployment${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo -e "Source directory: ${GREEN}$SOURCE_DIR${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
if [ "$DRY_RUN" = false ]; then
    echo -e "Hugging Face Space: ${GREEN}$HF_USERNAME/$SPACE_NAME${NC}"
fi
echo -e "Dry run: ${GREEN}$DRY_RUN${NC}"
echo ""

# Create output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Warning: Output directory already exists: $OUTPUT_DIR${NC}"
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Remove and continue? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Aborted${NC}"
            exit 0
        fi
    fi
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Copy files
echo -e "\n${BLUE}📁 Copying files...${NC}"

# Copy app.py
if [ -f "$SOURCE_DIR/app.py" ]; then
    cp "$SOURCE_DIR/app.py" "$OUTPUT_DIR/app.py"
    echo -e "  ${GREEN}✅${NC} app.py → app.py"
else
    echo -e "${RED}Error: app.py not found${NC}"
    exit 1
fi

# Copy requirements
if [ -f "$SOURCE_DIR/requirements_hf.txt" ]; then
    cp "$SOURCE_DIR/requirements_hf.txt" "$OUTPUT_DIR/requirements.txt"
    echo -e "  ${GREEN}✅${NC} requirements_hf.txt → requirements.txt"
else
    echo -e "${RED}Error: requirements_hf.txt not found${NC}"
    exit 1
fi

# Copy README
if [ -f "$SOURCE_DIR/README_HF.md" ]; then
    cp "$SOURCE_DIR/README_HF.md" "$OUTPUT_DIR/README.md"
    echo -e "  ${GREEN}✅${NC} README_HF.md → README.md"
else
    echo -e "${RED}Error: README_HF.md not found${NC}"
    exit 1
fi

# Copy runtime.txt
if [ -f "$SOURCE_DIR/runtime.txt" ]; then
    cp "$SOURCE_DIR/runtime.txt" "$OUTPUT_DIR/runtime.txt"
    echo -e "  ${GREEN}✅${NC} runtime.txt → runtime.txt"
else
    echo -e "  ${YELLOW}Warning: runtime.txt not found, skipping${NC}"
fi

# Copy audio_rag package
if [ -d "$SOURCE_DIR/audio_rag" ]; then
    cp -r "$SOURCE_DIR/audio_rag" "$OUTPUT_DIR/"
    echo -e "  ${GREEN}✅${NC} audio_rag/ → audio_rag/"
else
    echo -e "${RED}Error: audio_rag/ directory not found${NC}"
    exit 1
fi

# Copy config
mkdir -p "$OUTPUT_DIR/conf"
if [ -f "$SOURCE_DIR/conf/config_hf.yaml" ]; then
    cp "$SOURCE_DIR/conf/config_hf.yaml" "$OUTPUT_DIR/conf/"
    echo -e "  ${GREEN}✅${NC} conf/config_hf.yaml → conf/config_hf.yaml"
else
    echo -e "${RED}Error: conf/config_hf.yaml not found${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ All files copied successfully!${NC}"

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "\n${BLUE}=========================================================================${NC}"
    echo -e "${GREEN}✅ Dry run completed successfully!${NC}"
    echo -e "Files prepared in: ${GREEN}$OUTPUT_DIR${NC}"
    echo -e "\nTo deploy manually:"
    echo -e "  cd $OUTPUT_DIR"
    echo -e "  git init"
    echo -e "  git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/$SPACE_NAME"
    echo -e "  git add ."
    echo -e "  git commit -m 'Deploy Audio RAG demo'"
    echo -e "  git push -u origin main"
    echo -e "${BLUE}=========================================================================${NC}"
    exit 0
fi

# Confirm deployment
if [ "$SKIP_CONFIRM" = false ]; then
    echo -e "\n${BLUE}=========================================================================${NC}"
    echo -e "Ready to deploy to: ${GREEN}https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME${NC}"
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 0
    fi
fi

# Initialize git
echo -e "\n${BLUE}🔧 Initializing git repository...${NC}"
cd "$OUTPUT_DIR"

git init
echo -e "  ${GREEN}✅${NC} Git repository initialized"

git remote add origin "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo -e "  ${GREEN}✅${NC} Remote added: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Configure git
git config user.email "$HF_USERNAME@users.noreply.huggingface.co"
git config user.name "$HF_USERNAME"

# Add and commit
echo -e "\n${BLUE}📤 Committing and pushing to Hugging Face...${NC}"

git add .
echo -e "  ${GREEN}✅${NC} Files staged"

git commit -m "Deploy Audio RAG demo"
echo -e "  ${GREEN}✅${NC} Changes committed"

# Push with token
git remote set-url origin "https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Try to push to main branch
if git push -u origin main --force 2>&1; then
    echo -e "  ${GREEN}✅${NC} Pushed to Hugging Face Spaces!"
else
    echo -e "${YELLOW}Warning: Push to main failed, trying master branch...${NC}"
    if git push -u origin master --force 2>&1; then
        echo -e "  ${GREEN}✅${NC} Pushed to Hugging Face Spaces (master branch)!"
    else
        echo -e "${RED}Error: Failed to push${NC}"
        exit 1
    fi
fi

# Success message
echo -e "\n${BLUE}=========================================================================${NC}"
echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo -e "Your demo is now live at:"
echo -e "  ${GREEN}https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME${NC}"
echo -e "\nIt may take a few minutes to build and start the demo."
echo -e "Check the logs at:"
echo -e "  ${GREEN}https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME/logs${NC}"
echo -e "${BLUE}=========================================================================${NC}"
