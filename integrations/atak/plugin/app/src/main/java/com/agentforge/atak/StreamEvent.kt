package com.agentforge.atak

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@Serializable
data class StreamEvent(
    val type: String,
    val lat: Double? = null,
    val lon: Double? = null,
    val title: String? = null
) {
    companion object {
        fun parse(line: String): StreamEvent? {
            return try {
                val json = Json { ignoreUnknownKeys = true }
                json.decodeFromString(StreamEvent.serializer(), line)
            } catch (_: Exception) {
                null
            }
        }
    }
}
