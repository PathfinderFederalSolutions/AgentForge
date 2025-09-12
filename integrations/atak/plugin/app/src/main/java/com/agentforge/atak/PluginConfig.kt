package com.agentforge.atak

import kotlinx.serialization.Serializable

@Serializable
data class PluginConfig(
    val baseUrl: String,
    val eventsPath: String = "/events/stream",
    val pollIntervalSec: Long = 5,
    val mtls: MtlsConfig? = null,
    val sig: SigConfig? = null
)

@Serializable
data class MtlsConfig(
    val clientP12: String = "client.p12",
    val clientP12Password: String = "changeit",
    val caCert: String = "ca.crt"
)

@Serializable
data class SigConfig(
    val algo: String = "ed25519",
    val pubKey: String = ""
)
