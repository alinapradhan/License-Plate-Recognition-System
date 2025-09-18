#!/bin/bash

# License Plate Detection System Setup Script

set -e  # Exit on error

echo "🚗 License Plate Detection System Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Found Python $PYTHON_VERSION"
}

# Check if pip is available
check_pip() {
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        print_error "pip is required but not installed."
        exit 1
    fi
    print_status "Found pip"
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Python Dependencies"
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found. Installing core dependencies..."
        pip install ultralytics opencv-python easyocr fastapi uvicorn pillow numpy
    fi
    
    print_status "Python dependencies installed successfully!"
}

# Setup directory structure
setup_directories() {
    print_step "Setting Up Directory Structure"
    
    directories=(
        "data/raw"
        "data/processed" 
        "data/samples"
        "models/pretrained"
        "models/trained"
        "results/images"
        "results/videos"
        "logs"
        "uploads"
        "exported_models"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Setup dataset (optional)
setup_dataset() {
    print_step "Dataset Setup"
    
    echo "Do you want to download the Kaggle Car Plate Detection dataset? (requires Kaggle API setup)"
    echo "1) Yes, download dataset"
    echo "2) No, create sample structure only"
    read -p "Enter your choice (1-2): " choice
    
    case $choice in
        1)
            if command -v kaggle &> /dev/null; then
                print_status "Downloading Kaggle dataset..."
                python scripts/download_dataset.py --kaggle --setup-dirs
            else
                print_warning "Kaggle CLI not found. Install with: pip install kaggle"
                print_warning "Then setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
                python scripts/download_dataset.py --setup-dirs
            fi
            ;;
        2)
            print_status "Creating sample directory structure..."
            python scripts/download_dataset.py --setup-dirs --samples
            ;;
        *)
            print_warning "Invalid choice. Creating sample structure..."
            python scripts/download_dataset.py --setup-dirs
            ;;
    esac
}

# Test the installation
test_installation() {
    print_step "Testing Installation"
    
    print_status "Testing detector initialization..."
    python -c "
import sys
sys.path.append('src')
try:
    from models.detector import LicensePlateDetector
    detector = LicensePlateDetector()
    print('✓ Detector initialization successful!')
except Exception as e:
    print(f'✗ Error initializing detector: {e}')
    exit(1)
"
    
    print_status "Testing database setup..."
    python -c "
import sys
sys.path.append('src')
try:
    from database.models import DatabaseManager
    db = DatabaseManager()
    print('✓ Database setup successful!')
except Exception as e:
    print(f'✗ Error setting up database: {e}')
    exit(1)
"
}

# Setup development environment
setup_dev_environment() {
    print_step "Development Environment Setup"
    
    # Check if Node.js is available for React frontend
    if command -v node &> /dev/null; then
        print_status "Node.js found. Setting up React frontend..."
        cd frontend
        
        if [ ! -f "package.json" ]; then
            print_error "package.json not found in frontend directory"
            cd ..
            return 1
        fi
        
        if command -v npm &> /dev/null; then
            npm install
            print_status "React dependencies installed!"
        elif command -v yarn &> /dev/null; then
            yarn install
            print_status "React dependencies installed with Yarn!"
        else
            print_warning "Neither npm nor yarn found. Cannot install React dependencies."
        fi
        
        cd ..
    else
        print_warning "Node.js not found. React frontend setup skipped."
        print_warning "Install Node.js to enable the web interface: https://nodejs.org/"
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_step "Creating Startup Scripts"
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting License Plate Detection API..."
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
EOF
    
    chmod +x start_backend.sh
    print_status "Created start_backend.sh"
    
    # Frontend startup script
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🌐 Starting React Frontend..."
cd frontend
if command -v npm &> /dev/null; then
    npm start
elif command -v yarn &> /dev/null; then
    yarn start
else
    echo "Neither npm nor yarn found. Cannot start frontend."
    exit 1
fi
EOF
        
        chmod +x start_frontend.sh
        print_status "Created start_frontend.sh"
    fi
    
    # Complete startup script
    cat > start_app.sh << 'EOF'
#!/bin/bash
echo "🚗 Starting Complete License Plate Detection System..."

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start frontend if available
if [ -f "start_frontend.sh" ]; then
    ./start_frontend.sh &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
fi

echo "Backend started with PID: $BACKEND_PID"
echo ""
echo "🎉 Application is starting!"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:3000 (if React is setup)"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'kill $BACKEND_PID; kill $FRONTEND_PID 2>/dev/null; exit' INT
wait
EOF
    
    chmod +x start_app.sh
    print_status "Created start_app.sh"
}

# Print usage instructions
print_usage() {
    print_step "Usage Instructions"
    
    cat << 'EOF'

🎉 Setup completed successfully!

Quick Start:
1. Start the complete application:
   ./start_app.sh

2. Or start services individually:
   ./start_backend.sh    # Start FastAPI backend
   ./start_frontend.sh   # Start React frontend (if available)

3. Test image detection:
   python src/inference/detect_plates.py --image path/to/image.jpg

4. Test webcam detection:
   python src/inference/detect_plates.py --webcam

5. Train custom model:
   python src/training/train_detector.py --data data/processed/dataset.yaml

API Endpoints:
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

Docker Deployment:
- cd docker && docker-compose up -d

For more information, see README.md
EOF
}

# Main setup function
main() {
    print_status "Starting License Plate Detection System setup..."
    
    # System checks
    check_python
    check_pip
    
    # Setup
    setup_directories
    install_dependencies
    setup_dataset
    setup_dev_environment
    
    # Test installation
    test_installation
    
    # Create startup scripts
    create_startup_scripts
    
    # Print usage
    print_usage
    
    print_status "Setup completed successfully! 🎉"
}

# Parse command line arguments
SKIP_DATASET=false
SKIP_FRONTEND=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-dataset     Skip dataset download/setup"
            echo "  --skip-frontend    Skip React frontend setup"
            echo "  --quiet           Minimize output"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main setup
if [ "$SKIP_DATASET" = false ]; then
    main
else
    # Modified main without dataset setup
    print_status "Starting License Plate Detection System setup (dataset skipped)..."
    check_python
    check_pip
    setup_directories
    install_dependencies
    if [ "$SKIP_FRONTEND" = false ]; then
        setup_dev_environment
    fi
    test_installation
    create_startup_scripts
    print_usage
    print_status "Setup completed successfully! 🎉"
fi