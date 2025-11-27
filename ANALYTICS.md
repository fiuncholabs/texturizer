# Google Analytics Setup Guide

This application includes Google Analytics 4 (GA4) tracking to help you understand how users interact with your STL Texturizer.

---

## Setup Instructions

### Step 1: Create a Google Analytics Account

1. Go to [Google Analytics](https://analytics.google.com/)
2. Sign in with your Google account
3. Click **"Start measuring"** or **"Admin"** → **"Create Property"**
4. Fill in your property details:
   - Property name: "STL Texturizer" (or your preferred name)
   - Reporting time zone: Your time zone
   - Currency: Your currency
5. Click **"Next"**
6. Select industry category and business size
7. Select how you plan to use Google Analytics
8. Click **"Create"**
9. Accept the Terms of Service

### Step 2: Set Up a Data Stream

1. Click **"Web"** as your platform
2. Enter your website URL (e.g., `https://yourdomain.com`)
3. Enter a stream name (e.g., "STL Texturizer Website")
4. Click **"Create stream"**
5. **Copy your Measurement ID** - it will look like `G-XXXXXXXXXX`

### Step 3: Configure Your Application

1. **For local development**, create a `.env` file:
   ```bash
   cp .env.example .env
   ```

2. **Add your Measurement ID** to the `.env` file:
   ```bash
   GA_MEASUREMENT_ID=G-XXXXXXXXXX
   ```

3. **For production deployment**, set the environment variable:

   **DigitalOcean App Platform:**
   - Go to your app's Settings → Environment Variables
   - Add: `GA_MEASUREMENT_ID` = `G-XXXXXXXXXX`

   **Docker:**
   ```bash
   docker run -e GA_MEASUREMENT_ID=G-XXXXXXXXXX ...
   ```

   **Docker Compose:**
   ```yaml
   environment:
     - GA_MEASUREMENT_ID=G-XXXXXXXXXX
   ```

   **Render/Railway/Fly.io:**
   - Add environment variable in dashboard or CLI

4. **Restart your application** for changes to take effect

---

## What's Being Tracked

The application automatically tracks the following events:

### Pageviews
- Automatically tracked when users visit the site
- No configuration needed

### Custom Events

#### 1. **File Uploaded**
- **Event name:** `file_uploaded`
- **Parameters:**
  - `file_size_category`: `small` (<1MB), `medium` (1-10MB), or `large` (>10MB)
- **When triggered:** User selects an STL file to upload

#### 2. **Default Cube Selected**
- **Event name:** `default_cube_selected`
- **When triggered:** User checks the "Use default cube" checkbox

#### 3. **Processing Started**
- **Event name:** `processing_started`
- **Parameters:**
  - `input_type`: `file` or `cube`
  - `noise_type`: `classic`, `perlin`, `billow`, `ridged`, or `voronoi`
- **When triggered:** User clicks the "Process STL" button

#### 4. **Processing Success**
- **Event name:** `processing_success`
- **Parameters:**
  - `input_type`: `file` or `cube`
  - `noise_type`: Noise algorithm used
  - `processing_time`: Time in seconds
- **When triggered:** STL processing completes successfully

#### 5. **Processing Error**
- **Event name:** `processing_error`
- **Parameters:**
  - `input_type`: `file` or `cube`
  - `noise_type`: Noise algorithm attempted
  - `error_message`: Error description
- **When triggered:** STL processing fails

#### 6. **File Downloaded**
- **Event name:** `file_downloaded`
- **Parameters:**
  - `file_name`: Name of the downloaded file
- **When triggered:** User downloads the processed STL file

---

## Viewing Your Analytics

### Accessing Google Analytics

1. Go to [Google Analytics](https://analytics.google.com/)
2. Select your property ("STL Texturizer")
3. You'll see your dashboard with real-time data

### Key Reports to Check

#### 1. **Real-time Report**
- Path: Reports → Realtime
- Shows users currently on your site
- See events as they happen

#### 2. **Events Report**
- Path: Reports → Engagement → Events
- See all custom events (file_uploaded, processing_success, etc.)
- Click on any event to see its parameters

#### 3. **User Acquisition**
- Path: Reports → Acquisition → User acquisition
- See where your traffic comes from (direct, search, social, etc.)

#### 4. **Pages and Screens**
- Path: Reports → Engagement → Pages and screens
- See pageview data

### Creating Custom Reports

#### Track Most Popular Noise Types

1. Go to **Explore** in the left sidebar
2. Click **"Blank"** to create a new exploration
3. In Variables:
   - Add Dimension: `Event name`
   - Add Dimension: `noise_type` (custom parameter)
   - Add Metric: `Event count`
4. In Settings:
   - Rows: `noise_type`
   - Values: `Event count`
   - Filters: `Event name` = `processing_success`

#### Track Success Rate

1. Create a new exploration
2. Add these metrics:
   - `Event count` for `processing_started`
   - `Event count` for `processing_success`
   - `Event count` for `processing_error`
3. Calculate success rate: (successes / started) × 100

#### Track Processing Times

1. Create a new exploration
2. Add dimension: `Event name`
3. Add metric: Average of `processing_time` (custom parameter)
4. Filter by `Event name` = `processing_success`

---

## Privacy Considerations

### What Google Analytics Collects

- **Pageviews** and time on site
- **User location** (country, region, city)
- **Device information** (browser, OS, screen size)
- **Traffic source** (how users found your site)
- **Custom events** (as listed above)

### What It Doesn't Collect

- **Personal information** (names, emails, addresses)
- **STL file contents** or names
- **IP addresses** (anonymized by default in GA4)

### GDPR Compliance

For users in Europe, you may need to:

1. **Add a cookie consent banner**
   - Use a tool like [Cookiebot](https://www.cookiebot.com/) or [Cookie Consent](https://www.osano.com/cookieconsent)
   - Block GA tracking until user consents

2. **Update your privacy policy**
   - Disclose that you use Google Analytics
   - Explain what data is collected
   - Provide opt-out instructions

3. **Enable IP anonymization** (already done by default in GA4)

### Sample Privacy Policy Text

Add this to your privacy policy:

```
We use Google Analytics to understand how visitors use our website. Google Analytics collects information such as how often users visit our site, what pages they visit, and what other sites they used prior to coming to our site. We use this information to improve our website and services.

Google Analytics collects only the IP address assigned to you on the date you visit our site, rather than your name or other identifying information. We do not combine the information collected through Google Analytics with personally identifiable information.

For more information about Google Analytics, please visit: https://policies.google.com/privacy
```

---

## Disabling Analytics (Optional)

### For Development

To disable analytics during local development, simply don't set the `GA_MEASUREMENT_ID` environment variable. The tracking code will not load if the variable is not set.

### For Users

Users can disable Google Analytics by:
- Using browser extensions like uBlock Origin or Privacy Badger
- Enabling "Do Not Track" in their browser settings
- Installing the [Google Analytics Opt-out Browser Add-on](https://tools.google.com/dlpage/gaoptout)

---

## Troubleshooting

### Analytics Not Working

1. **Check if GA_MEASUREMENT_ID is set:**
   ```bash
   echo $GA_MEASUREMENT_ID
   ```

2. **Check browser console** for errors:
   - Open DevTools (F12)
   - Look for Google Analytics errors

3. **Verify Measurement ID format:**
   - Should be `G-XXXXXXXXXX` (starts with G-)
   - No quotes or extra spaces

4. **Check Real-time Report:**
   - Visit your site
   - Go to GA4 Real-time Report
   - Should see 1 active user (you)

### Events Not Showing Up

1. **Wait a few minutes** - Events can take 1-2 minutes to appear
2. **Check event names** - They're case-sensitive
3. **Verify in DebugView:**
   - Add `?debug_mode=true` to your URL
   - Go to Admin → DebugView in GA4
   - See events in real-time with full details

### Too Many Events

If you're getting charged for too many events (unlikely for a small site):

1. GA4 free tier includes **10 million events/month**
2. Remove less important events from the code
3. Consider sampling (only track percentage of users)

---

## Advanced Configuration

### Custom Dimensions

To add more tracking parameters:

1. **In your code**, add new parameters to events:
   ```javascript
   trackEvent('processing_success', {
       input_type: 'file',
       noise_type: 'perlin',
       processing_time: '12',
       mesh_size: 'large'  // New parameter
   });
   ```

2. **In Google Analytics:**
   - Go to Admin → Data display → Custom definitions
   - Click "Create custom dimension"
   - Dimension name: `Mesh Size`
   - Scope: Event
   - Event parameter: `mesh_size`
   - Click Save

### E-commerce Tracking

If you add payment features later, you can track purchases:

```javascript
// When user makes a donation
trackEvent('purchase', {
    transaction_id: 'T12345',
    value: 5.00,
    currency: 'USD',
    items: [{
        item_name: 'Donation'
    }]
});
```

### User Properties

Track persistent user traits:

```javascript
// Set user property (e.g., preferred noise type)
gtag('set', 'user_properties', {
    preferred_noise: 'perlin'
});
```

---

## Cost

Google Analytics 4 is **completely free** for standard use. Limits:

- **10 million events per month** (plenty for most sites)
- **25 custom dimensions** per property
- **50 custom metrics** per property

If you exceed these limits, you can upgrade to Google Analytics 360 (enterprise, $150k+/year), but this is unlikely for a personal project.

---

## Additional Resources

- [Google Analytics Documentation](https://support.google.com/analytics/)
- [GA4 Events Guide](https://developers.google.com/analytics/devguides/collection/ga4/events)
- [GA4 vs Universal Analytics](https://support.google.com/analytics/answer/11583528)
- [GDPR Compliance Guide](https://support.google.com/analytics/answer/9019185)

---

## Example Analytics Queries

### Most Popular Noise Types (Last 30 Days)

1. Go to Explore → Create new exploration
2. Add rows: `noise_type`
3. Add values: `Event count`
4. Add filter: `Event name` exactly matches `processing_success`

### Average Processing Time by Noise Type

1. Create new exploration
2. Rows: `noise_type`
3. Values: Average `processing_time`
4. Filter: `Event name` = `processing_success`

### File vs Cube Usage

1. Create new exploration
2. Rows: `input_type`
3. Values: `Event count`
4. Filter: `Event name` = `processing_started`

### Error Rate

1. Create calculated metric:
   - Numerator: Event count where `Event name` = `processing_error`
   - Denominator: Event count where `Event name` = `processing_started`
   - Formula: (Numerator / Denominator) × 100

---

## Questions?

If you have questions about Google Analytics setup:
- Check the [official documentation](https://support.google.com/analytics/)
- Search for tutorials on YouTube
- Ask in web development communities like Stack Overflow
