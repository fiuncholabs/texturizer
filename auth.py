"""
Google OAuth authentication module for STL Texturizer
Provides session-based authentication using Google OAuth 2.0
"""

import json
import requests
from flask import session, redirect, request, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient

# Initialize Flask-Login
login_manager = LoginManager()

# Simple in-memory user class (no database)
class User(UserMixin):
    """User model for session-based authentication"""
    def __init__(self, id_, name, email, profile_pic):
        self.id = id_
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

    @staticmethod
    def get(user_id):
        """Get user from session (no database lookup)"""
        if 'user' in session and session.get('user', {}).get('id') == user_id:
            user_data = session['user']
            return User(
                id_=user_data['id'],
                name=user_data['name'],
                email=user_data['email'],
                profile_pic=user_data['profile_pic']
            )
        return None

    @staticmethod
    def create(id_, name, email, profile_pic):
        """Create user and store in session"""
        user = User(id_, name, email, profile_pic)
        session['user'] = {
            'id': id_,
            'name': name,
            'email': email,
            'profile_pic': profile_pic
        }
        return user


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return User.get(user_id)


def get_google_provider_cfg(discovery_url):
    """Fetch Google's OpenID Connect configuration"""
    return requests.get(discovery_url).json()


def init_oauth(app):
    """Initialize OAuth for the Flask app"""
    # Only initialize if OAuth is enabled
    if not app.config.get('ENABLE_GOOGLE_AUTH', False):
        return None

    # Validate required configuration
    client_id = app.config.get('GOOGLE_CLIENT_ID')
    client_secret = app.config.get('GOOGLE_CLIENT_SECRET')

    if not client_id or not client_secret:
        app.logger.warning('Google OAuth enabled but credentials not configured')
        return None

    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    # Create OAuth client
    client = WebApplicationClient(client_id)

    return client


def login_route(app, client):
    """Handle login redirect to Google"""
    if not app.config.get('ENABLE_GOOGLE_AUTH'):
        return redirect(url_for('index'))

    # Get Google's provider configuration
    google_provider_cfg = get_google_provider_cfg(
        app.config['GOOGLE_DISCOVERY_URL']
    )
    authorization_endpoint = google_provider_cfg['authorization_endpoint']

    # Construct the authorization request
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=url_for('auth_callback', _external=True),
        scope=['openid', 'email', 'profile'],
    )

    return redirect(request_uri)


def callback_route(app, client):
    """Handle OAuth callback from Google"""
    if not app.config.get('ENABLE_GOOGLE_AUTH'):
        return redirect(url_for('index'))

    # Get authorization code from callback
    code = request.args.get('code')

    # Get Google's provider configuration
    google_provider_cfg = get_google_provider_cfg(
        app.config['GOOGLE_DISCOVERY_URL']
    )
    token_endpoint = google_provider_cfg['token_endpoint']

    # Prepare token request
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )

    # Exchange authorization code for access token
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(app.config['GOOGLE_CLIENT_ID'], app.config['GOOGLE_CLIENT_SECRET']),
    )

    # Parse the tokens
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Get user info from Google
    userinfo_endpoint = google_provider_cfg['userinfo_endpoint']
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # Verify email is verified by Google
    userinfo = userinfo_response.json()
    if not userinfo.get('email_verified'):
        return 'User email not verified by Google.', 400

    # Extract user information
    unique_id = userinfo['sub']
    users_email = userinfo['email']
    users_name = userinfo.get('name', users_email)
    picture = userinfo.get('picture', '')

    # Create and login user
    user = User.create(
        id_=unique_id,
        name=users_name,
        email=users_email,
        profile_pic=picture
    )

    login_user(user, remember=True)

    # Redirect back to main page
    return redirect(url_for('index'))


def logout_route():
    """Handle logout"""
    logout_user()

    # Clear session
    if 'user' in session:
        session.pop('user')

    return redirect(url_for('index'))
