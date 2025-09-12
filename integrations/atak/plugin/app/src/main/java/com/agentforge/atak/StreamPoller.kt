package com.agentforge.atak

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.Request
import java.nio.charset.Charset

class StreamPoller(
    private val http: MtlsHttpClient,
    private val cfg: PluginConfig,
    private val onEvent: (StreamEvent) -> Unit
) {
    private val scope = CoroutineScope(Dispatchers.IO)
    private var job: Job? = null

    fun start() {
        stop()
        job = scope.launch {
            while (isActive) {
                try {
                    val url = cfg.baseUrl.trimEnd('/') + cfg.eventsPath
                    val req = Request.Builder().url(url).get().build()
                    http.client.newCall(req).execute().use { resp ->
                        if (!resp.isSuccessful) throw RuntimeException("HTTP ${'$'}{resp.code}")
                        val body = resp.body ?: return@use
                        val bytes = body.bytes()
                        val text = String(bytes, Charset.forName("UTF-8"))
                        text.lineSequence().forEach { line ->
                            val evt = StreamEvent.parse(line)
                            if (evt != null) onEvent(evt)
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.w("AgentForgeAtak", "poll error: ${'$'}{e.message}")
                }
                delay((cfg.pollIntervalSec.takeIf { it > 0 } ?: 5) * 1000)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}
