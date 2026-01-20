package com.example.stressguard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class DataLoggerService : Service(), SensorEventListener {
    
    private lateinit var sensorManager: SensorManager
    private lateinit var wakeLock: PowerManager.WakeLock
    private lateinit var csvFile: File
    
    private var heartRateSensor: Sensor? = null
    private var accelerometer: Sensor? = null
    
    // Latest sensor values
    private var lastHR: Float = 0f
    private var lastAccX: Float = 0f
    private var lastAccY: Float = 0f
    private var lastAccZ: Float = 0f
    
    companion object {
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "stress_logger_channel"
        var isRunning = false
    }
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        
        // Wake lock to keep logging in background
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "StressLogger::WakeLock"
        )
        
        // Create CSV file
        csvFile = File("/sdcard/Android/data/com.example.stressguard/files/stress_log.csv")
        if (!csvFile.exists()) {
            csvFile.writeText("timestamp,datetime,hr,acc_x,acc_y,acc_z,acc_mag\n")
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification())
        
        // Register sensors
        heartRateSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        
        // Acquire wake lock
        if (!wakeLock.isHeld) {
            wakeLock.acquire(24 * 60 * 60 * 1000L) // 24 hours
        }
        
        isRunning = true
        
        // Start periodic logging (every 1 second)
        startPeriodicLogging()
        
        return START_STICKY
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        
        when (event.sensor.type) {
            Sensor.TYPE_HEART_RATE -> {
                lastHR = event.values[0]
                android.util.Log.d("StressGuard", "HR sensor event: ${event.values[0]} bpm")
            }
            Sensor.TYPE_ACCELEROMETER -> {
                lastAccX = event.values[0]
                lastAccY = event.values[1]
                lastAccZ = event.values[2]
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for basic logging
    }
    
    private fun startPeriodicLogging() {
        val timer = Timer()
        timer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                logCurrentData()
            }
        }, 0, 5000) // Log every 1 second
    }
    
    private fun logCurrentData() {
        try {
            val timestamp = System.currentTimeMillis()
            val datetime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                .format(Date(timestamp))
            
            // Calculate accelerometer magnitude
            val accMag = Math.sqrt(
                (lastAccX * lastAccX + lastAccY * lastAccY + lastAccZ * lastAccZ).toDouble()
            ).toFloat()
            
            // Write to CSV
            val line = "$timestamp,$datetime,$lastHR,$lastAccX,$lastAccY,$lastAccZ,$accMag\n"
            csvFile.appendText(line)
            
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Unregister sensors
        sensorManager.unregisterListener(this)
        
        // Release wake lock
        if (wakeLock.isHeld) {
            wakeLock.release()
        }
        
        isRunning = false
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Stress Logger",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Background stress data logging"
        }
        
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }
    
    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Stress Logger Running")
            .setContentText("Recording HR + ACC data...")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()
    }
}
