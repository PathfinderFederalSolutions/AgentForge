package com.agentforge.atak

import android.content.Context
import android.util.Base64
import kotlinx.serialization.json.Json
import java.io.InputStream
import java.security.MessageDigest

object PluginConfigLoader {
    fun load(context: Context): PluginConfig? {
        return try {
            val cfgBytes = readAsset(context, "config.json")
            // Optional signature verify (debug bypass)
            val sig = readAssetOrNull(context, "config.json.sig")
            if (sig != null) {
                val ok = verify(cfgBytes, sig)
                if (!ok) {
                    android.util.Log.w("AgentForgeAtak", "Config signature invalid; continuing in debug")
                }
            }
            Json { ignoreUnknownKeys = true }.decodeFromString(PluginConfig.serializer(), String(cfgBytes))
        } catch (e: Exception) {
            android.util.Log.e("AgentForgeAtak", "Failed to load config: ${e.message}")
            null
        }
    }

    private fun readAsset(ctx: Context, name: String): ByteArray {
        ctx.assets.open(name).use { return it.readBytes() }
    }

    private fun readAssetOrNull(ctx: Context, name: String): ByteArray? {
        return try { readAsset(ctx, name) } catch (_: Exception) { null }
    }

    private fun verify(data: ByteArray, sigB64: ByteArray): Boolean {
        // Placeholder for ed25519; simply check non-empty for debug build
        return sigB64.isNotEmpty() && data.isNotEmpty()
    }
}
