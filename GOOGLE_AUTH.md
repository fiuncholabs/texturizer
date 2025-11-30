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

### Local Development Testing (HTTP on localhost)

**Important**: OAuth normally requires HTTPS, but for local development testing over HTTP, the implementation includes a special flag that allows insecure transport.

#### Prerequisites
1. Google Cloud OAuth credentials configured with redirect URI: `http://localhost:8000/auth/callback`
2. `.env` file configured (see setup instructions above)

#### Environment Configuration for Development Testing

Your `.env` file should have:
```bash
# Enable development mode (allows HTTP OAuth for testing)
FLASK_ENV=development

# Enable Google OAuth
ENABLE_GOOGLE_AUTH=true

# Your Google OAuth credentials
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_DISCOVERY_URL=https://accounts.google.com/.well-known/openid-configuration

# Secret key for sessions
SECRET_KEY=dev-secret-key-for-testing-only-change-in-production
```

**Critical**: The `FLASK_ENV=development` setting enables the `OAUTHLIB_INSECURE_TRANSPORT` flag in `auth.py`, which allows OAuth to work over HTTP. This is **automatically disabled** in production.

#### Testing Steps

1. **Start the development server**:
   ```bash
   python3 app.py
   ```

2. **Verify OAuth is enabled** - Check the console output for:
   ```
   Google OAuth authentication enabled
   ```

3. **Open the application**: Navigate to `http://localhost:8000`

4. **Test sign-in flow**:
   - Click "Sign in with Google" button
   - You'll be redirected to Google's consent screen
   - Authorize the application
   - You should be redirected back to `http://localhost:8000/auth/callback`
   - Your name should appear in the UI with a "Sign Out" button

5. **Test authenticated session**:
   - Refresh the page - you should still be signed in
   - Process an STL file - functionality should work identically
   - Your session persists as long as the browser is open

6. **Test sign-out**:
   - Click "Sign Out" button
   - Session should be cleared
   - "Sign in with Google" button should appear again

7. **Test without authentication**:
   - Set `ENABLE_GOOGLE_AUTH=false` in `.env`
   - Restart the server
   - Verify no authentication UI appears
   - Verify all STL processing works identically

#### Troubleshooting Development Testing

**"insecure transport error"**
- Ensure `FLASK_ENV=development` is set in `.env`
- Restart the application after changing `.env`
- The insecure transport flag only works in development mode

**"redirect_uri_mismatch"**
- Verify your Google Cloud Console has `http://localhost:8000/auth/callback` (exact URL)
- Check for trailing slashes - they must match exactly

**Session not persisting**
- Ensure `SECRET_KEY` is set in `.env`
- Check that cookies are enabled in your browser
- Clear browser cookies and try again

### Production Deployment

**CRITICAL SECURITY REQUIREMENTS** for production:

#### 1. HTTPS is Mandatory

OAuth **will not work** over HTTP in production. You must have:
- Valid SSL/TLS certificate installed
- HTTPS enabled on your domain
- HTTP redirects to HTTPS

#### 2. Environment Configuration

Your production `.env` must have:
```bash
# Production environment (disables insecure transport)
FLASK_ENV=production

# Enable Google OAuth
ENABLE_GOOGLE_AUTH=true

# Production OAuth credentials
GOOGLE_CLIENT_ID=your-production-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-production-client-secret

# CRITICAL: Strong random secret key for session encryption
SECRET_KEY=your-strong-random-secret-key-here

# Discovery URL (usually same)
GOOGLE_DISCOVERY_URL=https://accounts.google.com/.well-known/openid-configuration
```

**Generate a strong SECRET_KEY**:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

#### 3. Google Cloud Console Configuration

1. **Authorized JavaScript origins**:
   - `https://yourdomain.com`

2. **Authorized redirect URIs**:
   - `https://yourdomain.com/auth/callback`

   ⚠️ Must be exact match, including `https://` and no trailing slash

3. **OAuth Consent Screen**:
   - Configure app name and logo
   - Add privacy policy URL
   - Add terms of service URL
   - Submit for verification if making app public

#### 4. Production Testing Checklist

Before going live, verify:

- [ ] `FLASK_ENV=production` set in environment
- [ ] HTTPS working correctly on domain
- [ ] OAuth credentials configured for production domain
- [ ] Redirect URIs match production URLs exactly
- [ ] `SECRET_KEY` is strong, random, and persistent across restarts
- [ ] `SECRET_KEY` is kept confidential (not in version control)
- [ ] OAuth consent screen fully configured
- [ ] Privacy policy and ToS accessible
- [ ] Test sign in flow over HTTPS
- [ ] Test sign out clears session properly
- [ ] Sessions persist across page refreshes
- [ ] Error pages display user-friendly messages
- [ ] Server logs don't expose sensitive information
- [ ] Rate limiting configured appropriately

#### 5. Deployment Commands

**Using Gunicorn (recommended)**:
```bash
gunicorn --config gunicorn.conf.py app:app
```

**Environment variables** can be set via:
- `.env` file (ensure it's not in git)
- System environment variables
- Container orchestration (Docker, Kubernetes)
- Platform-specific config (Heroku, AWS, etc.)

#### 6. Security Checklist

Production security requirements:

- [ ] HTTPS/TLS enabled with valid certificate
- [ ] `FLASK_ENV=production` (disables debug mode and insecure transport)
- [ ] Strong `SECRET_KEY` (32+ random bytes)
- [ ] `.env` file not committed to version control
- [ ] OAuth Client Secret kept confidential
- [ ] Rate limiting enabled (`RATELIMIT_ENABLED=true`)
- [ ] CORS configured appropriately
- [ ] Session timeout configured
- [ ] Error messages don't leak sensitive information
- [ ] Logs don't contain credentials or tokens

#### 7. How Insecure Transport Flag Works

The implementation in `auth.py` includes:

```python
# Allow insecure transport for local development (HTTP instead of HTTPS)
# WARNING: Only use this for local development, never in production!
if os.environ.get('FLASK_ENV') == 'development':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
```

**Key points**:
- Only activates when `FLASK_ENV=development`
- Automatically disabled in production (`FLASK_ENV=production`)
- Allows OAuth over HTTP for localhost testing
- Should **never** be manually enabled in production
- Google will reject OAuth over HTTP in production anyway

#### 8. Migrating from Development to Production

When deploying to production:

1. **Update `.env` file**:
   - Change `FLASK_ENV=development` → `FLASK_ENV=production`
   - Update `SECRET_KEY` to strong random value
   - Update OAuth credentials to production values

2. **Update Google Cloud Console**:
   - Add production domain to authorized origins
   - Add `https://yourdomain.com/auth/callback` to redirect URIs

3. **Enable HTTPS**:
   - Install SSL/TLS certificate
   - Configure web server (nginx, Apache, etc.)
   - Test HTTPS access

4. **Deploy and test**:
   - Deploy application with production configuration
   - Test OAuth flow over HTTPS
   - Verify sessions work correctly
   - Monitor logs for any errors

### Production Testing Checklist

Final verification before launch:

- [ ] OAuth works over HTTPS
- [ ] Sign in redirects to Google correctly
- [ ] Callback returns to production domain
- [ ] User info displays correctly after authentication
- [ ] Sessions persist across requests
- [ ] Sign out clears session completely
- [ ] Application works identically with/without authentication
- [ ] No errors in production logs
- [ ] Performance is acceptable under load

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
