package com.agentforge.atak

import java.io.BufferedInputStream
import java.io.InputStream
import java.security.KeyStore
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

object TlsUtil {
    fun addCertificateToKeyStore(keyStore: KeyStore, caStream: InputStream) {
        val cf = CertificateFactory.getInstance("X.509")
        val bis = BufferedInputStream(caStream)
        val cert = cf.generateCertificate(bis) as X509Certificate
        keyStore.setCertificateEntry("ca", cert)
    }

    fun systemDefaultTrustManager(): X509TrustManager {
        val tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
        tmf.init(null as KeyStore?)
        val tms = tmf.trustManagers
        return tms.filterIsInstance<X509TrustManager>().first()
    }
}
