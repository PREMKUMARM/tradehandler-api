# Google OAuth Setup Guide

## Backend Setup

### 1. Install Dependencies
```bash
pip install httpx PyJWT
```

### 2. Configure Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **Google+ API** or **Google Identity API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Configure OAuth consent screen:
   - User Type: External (or Internal for G Suite)
   - Application name: AlgoFeast AI
   - Authorized domains: your domain
   - Scopes: `email`, `profile`, `openid`
6. Create OAuth 2.0 Client ID:
   - Application type: Web application
   - Authorized redirect URIs:
     - `http://localhost:4200/auth/callback` (development)
     - `https://yourdomain.com/auth/callback` (production)

### 3. Set Environment Variables

Add to your `.env` file:

```env
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:4200/auth/callback
JWT_SECRET=your-secret-key-change-in-production
```

### 4. Database Migration

The users table will be created automatically on first run. It includes:
- `user_id` (primary key)
- `email` (unique)
- `name`
- `picture`
- `google_id` (unique)
- `created_at`
- `last_login`
- `is_active`

## Frontend Setup

### 1. Routes Configuration

Add auth routes to your main routing module:

```typescript
{
  path: 'auth',
  loadChildren: () => import('./modules/auth/auth.module').then(m => m.AuthModule)
}
```

### 2. Environment Configuration

Update `environment.ts`:

```typescript
export const environment = {
  api_host: 'http://localhost:8000',
  // ... other config
};
```

## API Endpoints

### 1. Initiate Google Login
```
GET /api/v1/auth/google/login
```
Redirects user to Google OAuth consent screen.

### 2. OAuth Callback
```
GET /api/v1/auth/google/callback?code=xxx
```
Handles Google OAuth callback, creates/updates user, generates JWT token.

### 3. Get Current User
```
GET /api/v1/auth/me
Headers:
  Authorization: Bearer <jwt_token>
```
Returns current authenticated user information.

### 4. Verify Token
```
GET /api/v1/auth/verify
Headers:
  Authorization: Bearer <jwt_token>
```
Verifies JWT token validity.

## User Flow

1. User clicks "Sign in with Google" on login page
2. Redirected to Google OAuth consent screen
3. User authorizes application
4. Google redirects to `/api/v1/auth/google/callback` with authorization code
5. Backend exchanges code for access token
6. Backend gets user info from Google
7. Backend creates/updates user in database
8. Backend generates JWT token
9. Backend redirects to frontend `/auth/callback?token=xxx&user_id=xxx`
10. Frontend stores token in localStorage
11. Frontend redirects to app

## Testing

### 1. Test Login Flow
1. Navigate to `/auth/login`
2. Click "Sign in with Google"
3. Complete Google OAuth flow
4. Should redirect to app with token stored

### 2. Test Token Verification
```bash
curl -X GET "http://localhost:8000/api/v1/auth/verify" \
  -H "Authorization: Bearer <your_jwt_token>"
```

### 3. Test Get Current User
```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <your_jwt_token>"
```

## Security Notes

1. **JWT Secret**: Change `JWT_SECRET` in production
2. **HTTPS**: Use HTTPS in production
3. **Token Expiration**: Tokens expire after 7 days (configurable)
4. **CORS**: Configure CORS properly for production
5. **Redirect URI**: Must match exactly in Google Console

## Troubleshooting

### Issue: "Google OAuth is not configured"
- Check `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`
- Restart backend server

### Issue: "redirect_uri_mismatch"
- Check redirect URI in Google Console matches exactly
- Check `GOOGLE_REDIRECT_URI` in `.env`

### Issue: Token not working
- Check JWT_SECRET matches between token generation and verification
- Check token expiration
- Verify token format: `Bearer <token>`

