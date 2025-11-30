# Google OAuth Authentication

This document explains how to set up and use optional Google OAuth authentication for the STL Texturizer application.

## Overview

The Google OAuth integration allows users to optionally sign in with their Google account. This feature is:
- **Fully optional** - The app works identically when authentication is disabled
- **Session-based** - Uses Flask sessions (no database required in v0.1)
- **Feature-flagged** - Enable/disable via environment variable

## Current Implementation (v0.1)

The current implementation provides:
- Session-based authentication using Google OAuth 2.0
- No persistent storage (sessions expire when browser closes)
- No impact on core texturizing functionality
- Clean graceful degradation when disabled

**Future versions** may add:
- User profile storage in database
- Processing queue management
- File history and saved presets

## Prerequisites

To enable Google OAuth, you need:

1. **Google Cloud Project** with OAuth 2.0 credentials
2. **OAuth Client ID and Secret** configured for web application
3. **Authorized redirect URIs** configured in Google Cloud Console

## Setup Instructions

### 1. Create Google Cloud OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services > Credentials**
4. Click **Create Credentials > OAuth client ID**
5. Select **Web application** as the application type
6. Configure the OAuth consent screen if prompted
7. Add authorized redirect URIs:
   - For development: `http://localhost:8000/auth/callback`
   - For production: `https://yourdomain.com/auth/callback`
8. Note your **Client ID** and **Client Secret**

**Official documentation**: https://developers.google.com/identity/protocols/oauth2/web-server

### 2. Configure Environment Variables

Edit your `.env` file and set the following:

```bash
# Enable Google OAuth
ENABLE_GOOGLE_AUTH=true

# OAuth credentials from Google Cloud Console
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret

# OAuth discovery URL (usually no need to change)
GOOGLE_DISCOVERY_URL=https://accounts.google.com/.well-known/openid-configuration

# Ensure SECRET_KEY is set for session management
SECRET_KEY=your-random-secret-key-here
```

**Security Notes**:
- Never commit your `.env` file to version control
- Use a strong, random SECRET_KEY in production
- Keep your GOOGLE_CLIENT_SECRET confidential

### 3. Install Dependencies

If you haven't already, install the required packages:

```bash
pip3 install -r requirements.txt
```

The OAuth-related dependencies are:
- `flask-login>=0.6.0` - User session management
- `requests>=2.28.0` - HTTP requests to Google APIs
- `oauthlib>=3.2.0` - OAuth protocol implementation
- `requests-oauthlib>=1.3.1` - OAuth for requests library

### 4. Restart the Application

```bash
# Development
python3 app.py

# Production
gunicorn --config gunicorn.conf.py app:app
```

## Usage

### When Authentication is Enabled

When `ENABLE_GOOGLE_AUTH=true`:

1. A "Sign in with Google" button appears in the UI
2. Users can click to authenticate with their Google account
3. After signing in, the user's name/email is displayed
4. A "Sign out" button allows logging out
5. Sessions persist until browser closes or explicit logout

### When Authentication is Disabled

When `ENABLE_GOOGLE_AUTH=false` (default):

1. No authentication UI is shown
2. App works identically to versions without auth
3. No session overhead or OAuth requests
4. Perfect for local/personal use

## Technical Architecture

### OAuth Flow

The Google OAuth 2.0 flow works as follows:

1. **User clicks "Sign in with Google"**
   - App redirects to Google's authorization page
   - User sees consent screen and approves

2. **Google redirects back to app**
   - Includes authorization code in callback URL
   - App receives code at `/auth/callback` endpoint

3. **App exchanges code for tokens**
   - Makes server-side request to Google
   - Receives access token and ID token

4. **App fetches user info**
   - Uses access token to get user profile
   - Stores in Flask session (no database)

5. **User is authenticated**
   - Session cookie tracks authentication state
   - User info available in templates/routes

### Session Management

- Uses Flask's built-in session management
- Session data stored in encrypted cookie
- Requires `SECRET_KEY` to be set
- Sessions expire on browser close (can be configured)

### Code Structure

OAuth implementation spans these files:

- `config.py` - OAuth configuration variables
- `.env.example` - Environment variable templates
- `GOOGLE_AUTH.md` - This documentation file

**Future implementation** will add:
- `auth.py` - OAuth routes and logic
- `templates/index.html` - Sign in/out buttons

## Security Considerations

### Best Practices

1. **Always use HTTPS in production**
   - OAuth requires secure redirect URIs
   - Configure SSL/TLS on your server

2. **Set strong SECRET_KEY**
   - Used to encrypt session cookies
   - Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"`

3. **Keep credentials confidential**
   - Never commit `.env` to git
   - Use environment variables in production
   - Rotate secrets if exposed

4. **Configure OAuth consent screen**
   - Add privacy policy and terms of service
   - Request minimal scopes needed
   - Use verification if publishing publicly

### OAuth Scopes

The implementation requests these Google scopes:
- `openid` - User identifier
- `email` - User's email address
- `profile` - Basic profile info (name, picture)

These are minimal scopes for basic authentication.

## Troubleshooting

### "Error: redirect_uri_mismatch"

**Cause**: The redirect URI in your request doesn't match what's configured in Google Cloud Console.

**Solution**:
1. Check your redirect URI in Google Cloud Console
2. Ensure it exactly matches: `http://localhost:8000/auth/callback` (dev) or your production URL
3. Include the protocol (`http://` or `https://`)
4. Check for trailing slashes

### "Error: invalid_client"

**Cause**: Client ID or Secret is incorrect.

**Solution**:
1. Double-check `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`
2. Ensure no extra whitespace in environment variables
3. Regenerate credentials in Google Cloud Console if needed

### Sessions not persisting

**Cause**: `SECRET_KEY` not set or app restarting.

**Solution**:
1. Set `SECRET_KEY` in `.env` file
2. Use the same SECRET_KEY across restarts
3. Check that cookies are enabled in browser

### "Sign in with Google" button not showing

**Cause**: `ENABLE_GOOGLE_AUTH` is false or not set.

**Solution**:
1. Set `ENABLE_GOOGLE_AUTH=true` in `.env`
2. Restart the application
3. Check browser console for JavaScript errors

## Testing

### Local Development Testing

1. Set `ENABLE_GOOGLE_AUTH=false` to test without authentication
2. Set `ENABLE_GOOGLE_AUTH=true` with test credentials to verify OAuth flow
3. Test sign in, processing STL files, sign out
4. Verify sessions persist across page refreshes
5. Verify sign out clears session

### Production Testing Checklist

- [ ] OAuth credentials configured for production domain
- [ ] HTTPS enabled and working
- [ ] Redirect URIs match production URLs
- [ ] SECRET_KEY is strong and persistent
- [ ] OAuth consent screen configured
- [ ] Privacy policy and ToS links working
- [ ] Sign in/out flow works correctly
- [ ] Sessions persist appropriately
- [ ] Error handling displays user-friendly messages

## Future Enhancements

Planned for future versions:

### Database Integration
- Store user profiles persistently
- Track processing history per user
- Save favorite settings/presets

### Processing Queue
- Queue multiple files for processing
- Email notifications when processing completes
- Priority processing for authenticated users

### File Management
- Store processed STL files temporarily
- Download history for authenticated users
- Share links for processed files

### Advanced Features
- Save and load parameter presets
- Compare before/after previews
- Usage analytics per user

## References

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google OAuth 2.0 for Web Server Apps](https://developers.google.com/identity/protocols/oauth2/web-server)
- [Flask-Login Documentation](https://flask-login.readthedocs.io/)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)

## Support

For issues related to:
- **OAuth setup**: Check Google Cloud Console documentation
- **App configuration**: See `.env.example` and `config.py`
- **General questions**: See `README.md` and `TROUBLESHOOTING.md`

---

*This feature is in active development. Documentation will be updated as implementation progresses.*
