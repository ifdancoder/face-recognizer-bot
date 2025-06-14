# Face Recognition System

This project consists of a Python microservice for face recognition and a Laravel backend for API management.

## Python Face Recognition Service

The Python service provides the following endpoints:

- `POST /train` - Train the model with a new face
  - Parameters:
    - `name`: Name of the person
    - `image`: Face image file
- `POST /recognize` - Recognize faces in an image
  - Parameters:
    - `image`: Image file containing faces
- `GET /known-faces` - Get list of all known faces

### Setup Python Service

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python face_recognition_service.py
```

The service will run on `http://localhost:8000`

## Laravel Backend (To be implemented)

The Laravel backend will provide:
- User authentication
- API endpoints for face recognition
- Database storage for face metadata
- Web interface for managing faces

### Setup Laravel

1. Create a new Laravel project:
```bash
composer create-project laravel/laravel face-recognition-api
```

2. Configure your database in `.env`

3. Run migrations:
```bash
php artisan migrate
```

4. Start the Laravel development server:
```bash
php artisan serve
```

## API Documentation

### Python Service Endpoints

#### Train Face
```http
POST /train
Content-Type: multipart/form-data

name: string
image: file
```

#### Recognize Face
```http
POST /recognize
Content-Type: multipart/form-data

image: file
```

#### Get Known Faces
```http
GET /known-faces
```

## Security Considerations

1. Always use HTTPS in production
2. Implement proper authentication
3. Validate and sanitize all inputs
4. Implement rate limiting
5. Secure storage of face encodings

## Dependencies

### Python
- face-recognition
- numpy
- opencv-python
- fastapi
- uvicorn
- python-multipart
- requests
- python-dotenv

### Laravel
- PHP 8.1+
- Composer
- MySQL/PostgreSQL 