package model

import (
	"fmt"
	"os"
	"path/filepath"
)

const (
	DataDir = "./data/users" // 用户数据根目录
)

// InitUserDataDir 初始化用户数据目录
func InitUserDataDir(username string) error {
	userDir := filepath.Join(DataDir, username)
	dirs := []string{
		filepath.Join(userDir, "source"),    // 原始音频文件
		filepath.Join(userDir, "processed"), // 处理后的音频文件
		filepath.Join(userDir, "models"),    // 训练的模型文件
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("创建目录失败 %s: %v", dir, err)
		}
	}

	return nil
}

// GetUserVoicePath 获取用户语音文件路径
func GetUserVoicePath(username, filename string) string {
	return filepath.Join(DataDir, username, "source", filename)
}

// GetUserProcessedPath 获取用户处理后的语音文件路径
func GetUserProcessedPath(username, filename string) string {
	return filepath.Join(DataDir, username, "processed", filename)
}

// GetUserModelPath 获取用户模型文件路径
func GetUserModelPath(username, filename string) string {
	return filepath.Join(DataDir, username, "models", filename)
}

// EnsureUserDir 确保用户目录存在
func EnsureUserDir(username string) error {
	return InitUserDataDir(username)
}
