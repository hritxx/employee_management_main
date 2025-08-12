#!/bin/bash
# Script to start the employee management application with different profiles

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}====== Aganitha Employee Management System ======${NC}"
echo

# Function to clean up previous containers
cleanup() {
    echo -e "${YELLOW}Stopping any existing containers...${NC}"
    docker-compose -f docker-compose-consolidated.yml down 2>/dev/null || true
    echo
}

# Function to display help
show_help() {
    echo -e "Usage: $0 [OPTION]"
    echo
    echo -e "Options:"
    echo -e "  --standard      Start the standard deployment (default)"
    echo -e "  --with-backup   Start with backup functionality enabled"
    echo -e "  --down          Stop all containers"
    echo -e "  --help          Display this help and exit"
    echo
    echo -e "Examples:"
    echo -e "  $0                   # Start standard deployment"
    echo -e "  $0 --with-backup     # Start with backup functionality"
    echo -e "  $0 --down            # Stop all containers"
    echo
}

# Handle command line arguments
case "$1" in
    --standard|"")
        cleanup
        echo -e "${GREEN}Starting standard deployment...${NC}"
        docker-compose -f docker-compose-consolidated.yml up -d
        echo
        echo -e "${GREEN}Application started successfully!${NC}"
        echo -e "Web interface available at: ${BLUE}http://localhost:8501${NC}"
        echo -e "To view logs, run: ${YELLOW}docker-compose -f docker-compose-consolidated.yml logs -f${NC}"
        ;;
    --with-backup)
        cleanup
        echo -e "${GREEN}Starting deployment with backup functionality...${NC}"
        docker-compose -f docker-compose-consolidated.yml --profile backup up -d
        echo
        echo -e "${GREEN}Application started successfully with backup scheduler!${NC}"
        echo -e "Web interface available at: ${BLUE}http://localhost:8501${NC}"
        echo -e "To view logs, run: ${YELLOW}docker-compose -f docker-compose-consolidated.yml logs -f${NC}"
        echo -e "Backup schedule: ${YELLOW}Weekly on Sundays at 2:00 AM${NC}"
        ;;
    --down)
        echo -e "${YELLOW}Stopping all containers...${NC}"
        docker-compose -f docker-compose-consolidated.yml down
        echo -e "${GREEN}All containers stopped successfully.${NC}"
        ;;
    --help)
        show_help
        ;;
    *)
        echo -e "${YELLOW}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac
