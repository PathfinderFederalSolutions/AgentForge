package com.agentforge.atak

import android.content.Context
import okhttp3.OkHttpClient
import java.io.InputStream
import java.security.KeyStore
import java.security.SecureRandom
import javax.net.ssl.KeyManagerFactory
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

class MtlsHttpClient(context: Context, cfg: PluginConfig) {
    val client: OkHttpClient

    init {
        val mtls = cfg.mtls ?: MtlsConfig()
        // Load client.p12 and ca.crt from assets based on config
        val assets = context.assets
        val p12 = safeOpen(assets, mtls.clientP12)
        val ca = safeOpen(assets, mtls.caCert)

        val kmf = buildKmf(p12, (mtls.clientP12Password).toCharArray())
        val tmf = buildTmf(ca)
        val x509Tm = tmf.trustManagers.filterIsInstance<X509TrustManager>().first()
        val sslContext = SSLContext.getInstance("TLS")
        sslContext.init(kmf.keyManagers, arrayOf(x509Tm), SecureRandom())

        client = OkHttpClient.Builder()
            .sslSocketFactory(sslContext.socketFactory, x509Tm)
            .build()
    }

    private fun safeOpen(assets: android.content.res.AssetManager, name: String): InputStream {
        return assets.open(name)
    }

    private fun buildKmf(p12Stream: InputStream, password: CharArray): KeyManagerFactory {
        val ks = KeyStore.getInstance("PKCS12")
        ks.load(p12Stream, password)
        val kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm())
        kmf.init(ks, password)
        return kmf
    }

    private fun buildTmf(caStream: InputStream): TrustManagerFactory {
        val ks = KeyStore.getInstance(KeyStore.getDefaultType())
        ks.load(null)
        TlsUtil.addCertificateToKeyStore(ks, caStream)
        val tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
        tmf.init(ks)
        return tmf
    }
}
