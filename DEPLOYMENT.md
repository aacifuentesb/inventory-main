# Supply Chain Management Application Deployment Guide

## Application Overview

This is a Supply Chain Management application built with Python and Streamlit. The application provides inventory management, forecasting, and optimization capabilities. Here's a breakdown of the main components:

- **Frontend**: Built with Streamlit
- **Core Features**:
  - Inventory Management (`inventory.py`)
  - Demand Forecasting (`forecast.py`)
  - SKU Management (`sku.py`)
  - Supply Chain Optimization (`optimization.py`)
- **Data Management**: Uses Excel files for data storage and manipulation

## System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB recommended)
- 50GB storage space
- Windows/Linux/macOS compatible

## Dependencies

The application requires the following Python packages:
- numpy
- scipy
- pandas
- plotly
- streamlit
- statsmodels
- statsforecast
- stqdm
- openpyxl

## Local Deployment Steps

### 1. Environment Setup

1. Install Python 3.8 or higher if not already installed
   ```bash
   # On Windows:
   # Download and install from https://www.python.org/downloads/
   
   # On Linux:
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   
   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 2. Application Configuration

1. Clone or copy all application files to your server
2. Ensure all Excel files are in the root directory:
   - data.xlsx
   - df_original.xlsx
   - df_reducida.xlsx
   - df_200.xlsx

### 3. Running the Application

1. Navigate to the application directory
   ```bash
   cd /path/to/application
   ```

2. Start the Streamlit server
   ```bash
   streamlit run Home.py
   ```

3. The application will be available at `http://localhost:8501` by default

### 4. Production Deployment Options

For production deployment, consider the following options:

#### Option 1: Using Streamlit's Built-in Server (Basic)
```bash
streamlit run Home.py --server.port 80 --server.address 0.0.0.0
```

#### Option 2: Using Nginx as a Reverse Proxy (Recommended)

1. Install Nginx
   ```bash
   # On Ubuntu/Debian
   sudo apt install nginx
   ```

2. Create Nginx configuration:
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header Host $host;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```

3. Run the application with supervisor or systemd for process management

#### Option 3: Using Docker (Recommended for Production)

Docker provides a consistent and isolated environment for running your application. Here's how to deploy using Docker:

1. Create a `Dockerfile` in your project root:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       software-properties-common \
       net-tools \
       && rm -rf /var/lib/apt/lists/*

   # Copy the application files
   COPY . .

   # Install Python dependencies
   RUN pip3 install -r requirements.txt

   # Expose Streamlit's default port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   # Set Streamlit configuration
   ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ENV STREAMLIT_SERVER_PORT=8501
   ENV STREAMLIT_SERVER_HEADLESS=true

   # Start the application
   ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Create a `docker-compose.yml` for production deployment:
   ```yaml
   version: '3'

   services:
     app:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - ./data:/app/data
         - .:/app
       restart: always
       environment:
         - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5
         - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
         - STREAMLIT_SERVER_ADDRESS=0.0.0.0
         - STREAMLIT_SERVER_PORT=8501
         - STREAMLIT_SERVER_HEADLESS=true
       networks:
         - streamlit_network

   networks:
     streamlit_network:
       driver: bridge
   ```

3. Deploy the application:

   a. First-time deployment:
   ```bash
   # Build the container
   docker-compose build --no-cache

   # Start the application
   docker-compose up -d
   ```

   b. Access the application:
   - Windows/Linux: `http://localhost:8501`
   - macOS: `http://127.0.0.1:8501`

   c. Stop the application:
   ```bash
   docker-compose down
   ```

#### Docker Deployment Notes:

1. Volume Mounts:
   - `./data:/app/data`: Persists Excel files
   - `.:/app`: Mounts the entire application for development

2. Environment Variables:
   - `STREAMLIT_SERVER_ADDRESS`: Binds to all network interfaces
   - `STREAMLIT_SERVER_PORT`: Sets the application port
   - `STREAMLIT_SERVER_HEADLESS`: Enables headless mode
   - `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`: Sets maximum file upload size
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Disables usage statistics

3. Networking:
   - Uses bridge network for container isolation
   - Exposes port 8501 for web access
   - Health check ensures application availability

#### Docker Troubleshooting Guide:

1. Cannot access the application:
   ```bash
   # Check if container is running
   docker ps

   # View container logs
   docker-compose logs -f

   # Verify port mapping
   docker port $(docker ps -q)
   ```

2. Container fails to start:
   ```bash
   # Rebuild without cache
   docker-compose build --no-cache

   # Check for port conflicts
   netstat -ano | findstr :8501    # Windows
   lsof -i :8501                   # Linux/macOS
   ```

3. Docker Desktop specific:
   - Ensure Docker Desktop is running
   - Check WSL 2 backend is enabled (Windows)
   - Verify container status in Docker Desktop dashboard

4. Access URLs to try:
   - `http://localhost:8501`
   - `http://127.0.0.1:8501`
   - `http://192.168.99.100:8501` (Docker Toolbox)

5. Resource Issues:
   - Check Docker Desktop resource allocation
   - Monitor container resources:
     ```bash
     docker stats
     ```

#### Docker Security Best Practices:

1. Image Security:
   - Use specific version tags instead of `latest`
   - Regularly update base images
   - Scan images for vulnerabilities

2. Runtime Security:
   - Never run containers as root
   - Implement resource limits
   - Use read-only file systems where possible

3. Data Security:
   - Use Docker secrets for sensitive data
   - Encrypt volume data at rest
   - Regular backup of persistent volumes

4. Network Security:
   - Use custom bridge networks
   - Implement network policies
   - Configure TLS for Docker daemon

### 5. Maintenance

1. Regular Updates:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. Monitor system resources:
   - CPU usage
   - Memory consumption
   - Disk space
   - Network bandwidth

3. Backup Strategy:
   - Regular backups of Excel files
   - System configuration backups
   - Database backups if applicable

## Troubleshooting

Common issues and solutions:

1. **Port already in use**
   ```bash
   # Change port
   streamlit run Home.py --server.port 8502
   ```

2. **Memory Issues**
   - Increase system RAM
   - Monitor memory usage
   - Consider data optimization

3. **Excel File Issues**
   - Ensure proper file permissions
   - Check file path configurations
   - Verify Excel file format compatibility

## Support

For technical support or questions:
1. Check the application logs
2. Review error messages in the console
3. Contact the development team

## Performance Optimization

1. Configure Streamlit's memory management:
   ```bash
   export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5
   ```

2. Optimize Excel file sizes
3. Implement caching where appropriate
4. Monitor and adjust server resources as needed

---

**Note**: This deployment guide assumes a basic server setup. Additional configuration might be needed based on specific server requirements or company policies. 