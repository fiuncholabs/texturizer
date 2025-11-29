"""
Configuration management for the STL Texturizer application
Supports both development and production environments
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # File upload
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB default
    UPLOAD_TIMEOUT = int(os.environ.get('UPLOAD_TIMEOUT', 300))  # 5 minutes

    # Processing
    PROCESSING_TIMEOUT = int(os.environ.get('PROCESSING_TIMEOUT', 600))  # 10 minutes
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 1))  # Number of concurrent processing jobs

    # Output constraints
    MAX_OUTPUT_TRIANGLES = int(os.environ.get('MAX_OUTPUT_TRIANGLES', 20_000_000))  # 20 million triangles
    MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', 4096))  # 4GB
    MAX_OUTPUT_FILE_SIZE_MB = int(os.environ.get('MAX_OUTPUT_FILE_SIZE_MB', 500))  # 500MB

    # Rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '10 per minute')
    RATELIMIT_PROCESSING = os.environ.get('RATELIMIT_PROCESSING', '3 per minute')

    # CORS
    CORS_ENABLED = os.environ.get('CORS_ENABLED', 'false').lower() == 'true'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', None)  # None = stdout only

    # Health check
    HEALTH_CHECK_ENABLED = True

    # Analytics
    GA_MEASUREMENT_ID = os.environ.get('GA_MEASUREMENT_ID', None)  # Google Analytics Measurement ID

    # Default object
    DEFAULT_OBJECT_TYPE = os.environ.get('DEFAULT_OBJECT_TYPE', 'cube').lower()  # 'cube' or 'custom'
    DEFAULT_OBJECT_SIZE = float(os.environ.get('DEFAULT_OBJECT_SIZE', 20))  # Size in mm


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    RATELIMIT_ENABLED = True

    # Ensure secret key is set in production
    @classmethod
    def init_app(cls, app):
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            import warnings
            warnings.warn(
                'SECRET_KEY not set! Using default. Set SECRET_KEY environment variable.',
                RuntimeWarning
            )


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    RATELIMIT_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
