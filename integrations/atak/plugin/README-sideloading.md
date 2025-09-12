# Sideloading the AgentForge ATAK Plugin (Debug)

Prereqs
- Android SDK + Android Studio or command-line tools
- Java 17
- A device/emulator with Google APIs image (SDK 34 recommended)

Steps
1. Build the APK
   - From project root of the plugin: `./gradlew :app:assembleDebug`
   - Output: `app/build/outputs/apk/debug/app-debug.apk`
2. Install on device
   - Enable developer options and USB debugging
   - `adb install -r app/build/outputs/apk/debug/app-debug.apk`
3. Configure
   - Place `client.p12`, `ca.crt`, `config.json`, and optional `config.json.sig` in `integrations/atak/plugin/config/`
   - Rebuild to include them as assets
4. Verify
   - Launch the app, confirm mock marker appears
   - If a test server is running at `https://10.0.2.2:8443/events/stream` with mTLS, streamed markers will render

## Build (debug) on macOS

- Ensure Java 17 is active (e.g., using jenv or /usr/libexec/java_home)
- From `integrations/atak/plugin/` run:

```
./gradlew :app:assembleDebug
```

Artifacts:
- `app/build/outputs/apk/debug/app-debug.apk`

Install to emulator:

```
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Verification:
- Launch "AgentForge ATAK" app
- You should see a map centered on SF and a "Mock Marker"
- If a dev server is running at `https://10.0.2.2:8443/events/stream` with mTLS, streamed markers will render

Notes:
- Config and certs are bundled from `app/src/main/assets` and `../config` (plugin/config)
- Place `client.p12` and `ca.crt` in `integrations/atak/plugin/config/` for local testing

Uninstall
- `adb uninstall com.agentforge.atak`
