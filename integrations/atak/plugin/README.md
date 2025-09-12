# ATAK Plugin Skeleton

This is a minimal Android plugin skeleton that polls an `/events/stream` endpoint using mTLS client authentication and renders markers on a map. It includes a simple config mechanism via a signed JSON file located in `integrations/atak/plugin/config/`.

Notes
- This skeleton uses an embedded map (osmdroid) to render a mock marker and any markers received from the event stream.
- mTLS is implemented with OkHttp using a client PKCS#12 (`client.p12`) and a CA certificate (`ca.crt`).
- Config is loaded from `config/config.json` at build time (copied into app assets). A detached signature `config.json.sig` can be verified via Ed25519 in release builds; debug bypass is allowed by default.

Directory tree
- integrations/atak/plugin/
  - config/
    - config.json (signed)
    - config.json.sig (detached signature)
    - ca.crt (PEM)
    - client.p12 (PKCS#12)
  - app/ (Android module)
  - build files (Gradle)

Build (Debug)
1. Ensure you have Android SDK installed and Java 17 available.
2. From the `integrations/atak/plugin` folder, build debug:
   - With Android Studio: Open this folder as a project and run the `app` configuration.
   - Or via terminal: `./gradlew :app:assembleDebug`
3. Output APK: `app/build/outputs/apk/debug/app-debug.apk`.

Sideload on device/emulator
- Enable developer mode and USB debugging on your device.
- Install the debug APK:
  - `adb install -r app/build/outputs/apk/debug/app-debug.apk`
- Launch the app. A mock marker should render at startup; markers from the event stream will appear as they arrive.

Config
- Place files in `integrations/atak/plugin/config/`:
  - `config.json`: settings (see example below)
  - `config.json.sig`: detached signature bytes (Ed25519, Base64)
  - `client.p12`: client certificate + key for mTLS
  - `ca.crt`: CA certificate to validate the server
- On build, these are bundled into application assets.

Example `config.json`
{
  "baseUrl": "https://10.0.2.2:8080",
  "eventsPath": "/events/stream",
  "pollIntervalSec": 5,
  "mtls": {
    "clientP12": "client.p12",
    "clientP12Password": "changeit",
    "caCert": "ca.crt"
  },
  "sig": {
    "algo": "ed25519",
    "pubKey": "<base64-public-key>"
  }
}

Signing
- Generate a signature of `config.json` using your Ed25519 private key and store it in `config.json.sig` (Base64). Example (OpenSSL 3):
  - Generate keypair: `openssl genpkey -algorithm ED25519 -out ed25519.key`
  - Extract public: `openssl pkey -in ed25519.key -pubout -out ed25519.pub`
  - Sign: `openssl pkeyutl -sign -inkey ed25519.key -rawin -in config.json -out config.sig`
  - Base64: `base64 -i config.sig > config.json.sig`
  - Base64 public key for `pubKey`: `openssl base64 -in ed25519.pub -A`
- Debug builds bypass signature failure (log a warning). Release builds can be configured to require a valid signature.

Event stream format
- The endpoint should return JSON Lines (one JSON object per line). Minimal schema:
  { "type": "marker", "lat": 37.7749, "lon": -122.4194, "title": "Test" }

mTLS expectations
- `client.p12` contains the client certificate and private key for authentication.
- `ca.crt` is the CA certificate used to validate the server.

Adapting to real ATAK plugin packaging
- This skeleton focuses on map rendering and network integration. To integrate with ATAK’s plugin APIs, substitute the Map rendering with ATAK MapCore components and wire into the ATAK plugin lifecycle. The network/mTLS/config code can be reused as-is.

Troubleshooting
- If SSL fails, verify `client.p12` password and that the server trusts the client cert.
- On emulator, `10.0.2.2` maps to host localhost.
- Use `adb logcat | grep AgentForgeAtak` to see logs.
