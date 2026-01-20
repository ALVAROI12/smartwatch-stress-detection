package com.example.stressguard

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {
    
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var markStressButton: Button
    
    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            statusText.text = "✓ Permissions granted\nReady to log"
        } else {
            statusText.text = "✗ Need permissions\nTap Start again"
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        android.util.Log.d("StressGuard", "MainActivity onCreate started")
        
        statusText = findViewById(R.id.statusText)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        markStressButton = findViewById(R.id.markStressButton)
        
        android.util.Log.d("StressGuard", "Views initialized")
        
        startButton.setOnClickListener { 
            android.util.Log.d("StressGuard", "START button clicked!")
            startLogging() 
        }
        stopButton.setOnClickListener { 
            android.util.Log.d("StressGuard", "STOP button clicked!")
            stopLogging() 
        }
        markStressButton.setOnClickListener { 
            android.util.Log.d("StressGuard", "MARK button clicked!")
            markStressEvent() 
        }
        
        android.util.Log.d("StressGuard", "Listeners set")
    }
    
    private fun startLogging() {
        android.util.Log.d("StressGuard", "startLogging() called")
        
        val permissions = arrayOf(
            Manifest.permission.BODY_SENSORS,
        )
        
        android.util.Log.d("StressGuard", "Checking permissions...")
        val needsPermission = permissions.any {
            val granted = ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            android.util.Log.d("StressGuard", "Permission $it: granted=$granted")
            !granted
        }
        
        android.util.Log.d("StressGuard", "needsPermission=$needsPermission")
        
        if (needsPermission) {
            android.util.Log.d("StressGuard", "Requesting permissions...")
            requestPermissions.launch(permissions)
            return
        }
        
        android.util.Log.d("StressGuard", "Starting service...")
        try {
            val intent = Intent(this, DataLoggerService::class.java)
            ContextCompat.startForegroundService(this, intent)
            android.util.Log.d("StressGuard", "Service start command sent")
        } catch (e: Exception) {
            android.util.Log.e("StressGuard", "Failed to start service", e)
        }
        
        statusText.text = "🟢 LOGGING...\n\nData saved to:\n/sdcard/Android/data/com.example.stressguard/files/stress_log.csv"
        updateUI()
        android.util.Log.d("StressGuard", "UI updated")
    }
    
    private fun stopLogging() {
        val intent = Intent(this, DataLoggerService::class.java)
        stopService(intent)
        
        val file = File("/sdcard/Android/data/com.example.stressguard/files/stress_log.csv")
        val lines = if (file.exists()) file.readLines().size else 0
        
        statusText.text = "⭕ Stopped\n\nRecorded: $lines samples\nFile: /sdcard/Android/data/com.example.stressguard/files/stress_log.csv"
        updateUI()
    }
    
    private fun markStressEvent() {
        // Append marker to CSV
        try {
            val file = File("/sdcard/Android/data/com.example.stressguard/files/stress_log.csv")
            if (file.exists()) {
                file.appendText("${System.currentTimeMillis()},STRESS_MARKER,,,,,\n")
                statusText.text = "✓ Stress event marked!"
            }
        } catch (e: Exception) {
            statusText.text = "✗ Error marking event"
        }
    }
    
    private fun updateUI() {
        // Simple UI state management
        val isRunning = DataLoggerService.isRunning
        startButton.isEnabled = !isRunning
        stopButton.isEnabled = isRunning
        markStressButton.isEnabled = isRunning
    }
    
    override fun onResume() {
        super.onResume()
        updateUI()
    }
}
