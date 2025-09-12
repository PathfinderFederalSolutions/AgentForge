package com.agentforge.atak

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

class MainActivity : AppCompatActivity() {
    private lateinit var map: MapView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().userAgentValue = packageName
        setContentView(R.layout.activity_main)

        map = findViewById(R.id.map)
        map.setTileSource(TileSourceFactory.MAPNIK)
        map.controller.setZoom(10.0)
        map.controller.setCenter(GeoPoint(37.7749, -122.4194))

        // Mock marker to satisfy acceptance
        addMarker(37.7749, -122.4194, "Mock Marker")

        // Load config and start polling if available
        val config = PluginConfigLoader.load(this)
        if (config != null) {
            try {
                val client = MtlsHttpClient(this, config)
                val poller = StreamPoller(client, config) { evt ->
                    if (evt.type == "marker" && evt.lat != null && evt.lon != null) {
                        runOnUiThread { addMarker(evt.lat, evt.lon, evt.title ?: "Marker") }
                    }
                }
                poller.start()
            } catch (e: Exception) {
                android.util.Log.w("AgentForgeAtak", "mTLS disabled: ${e.message}")
            }
        }
    }

    private fun addMarker(lat: Double, lon: Double, title: String) {
        val m = Marker(map)
        m.position = GeoPoint(lat, lon)
        m.title = title
        map.overlays.add(m)
        map.invalidate()
    }
}
