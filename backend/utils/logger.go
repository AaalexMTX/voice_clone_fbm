package utils

import (
	"os"
	"path/filepath"
	"time"

	log "github.com/sirupsen/logrus"
)

// InitLogger 初始化日志配置
func InitLogger() {
	// 设置日志格式为JSON
	log.SetFormatter(&log.TextFormatter{
		TimestampFormat: "2006-01-02 15:04:05",
		FullTimestamp:   true,
	})

	// 设置日志输出到标准输出（默认的）
	log.SetOutput(os.Stdout)

	// 设置日志级别
	log.SetLevel(log.InfoLevel)

	// 创建日志目录
	logDir := "logs"
	if err := os.MkdirAll(logDir, 0755); err != nil {
		log.Fatalf("无法创建日志目录: %v", err)
	}

	// 同时将日志写入文件
	logFileName := filepath.Join(logDir, time.Now().Format("2006-01-02")+".log")
	logFile, err := os.OpenFile(logFileName, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		log.SetOutput(logFile)
	} else {
		log.Errorf("无法写入日志文件: %v", err)
	}
}

// LogInfo 记录信息日志
func LogInfo(format string, args ...interface{}) {
	log.Infof(format, args...)
}

// LogError 记录错误日志
func LogError(format string, args ...interface{}) {
	log.Errorf(format, args...)
}

// LogWarning 记录警告日志
func LogWarning(format string, args ...interface{}) {
	log.Warnf(format, args...)
}

// LogDebug 记录调试日志
func LogDebug(format string, args ...interface{}) {
	log.Debugf(format, args...)
}

// LogFatal 记录致命错误日志
func LogFatal(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}
