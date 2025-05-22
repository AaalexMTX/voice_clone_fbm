package utils

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	log "github.com/sirupsen/logrus"
)

var (
	// 获取项目根目录的绝对路径
	currentDir, _  = os.Getwd()
	ProjectRoot, _ = filepath.Abs(filepath.Join(currentDir, ".."))
	DataDir        = filepath.Join(ProjectRoot, "data")
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
			LogError("创建目录失败 %s: %v", dir, err)
			return fmt.Errorf("创建目录失败 %s: %v", dir, err)
		}
	}

	log.Infof("已创建用户目录: %s", userDir)
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

// FileExists 检查文件是否存在
func FileExists(filepath string) bool {
	_, err := os.Stat(filepath)
	return !os.IsNotExist(err)
}

// CreateDirIfNotExist 如果目录不存在则创建
func CreateDirIfNotExist(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return os.MkdirAll(dir, 0755)
	}
	return nil
}

// DeleteFile 删除文件
func DeleteFile(filepath string) error {
	if FileExists(filepath) {
		return os.Remove(filepath)
	}
	return nil
}

// GetFileSize 获取文件大小
func GetFileSize(filepath string) (int64, error) {
	info, err := os.Stat(filepath)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// GetAudioDuration 获取音频文件的时长（秒）
func GetAudioDuration(filePath string) (float64, error) {
	// 检查ffprobe命令是否可用
	if _, err := exec.LookPath("ffprobe"); err != nil {
		LogWarning("ffprobe命令不可用，无法获取音频时长: %v", err)
		return 0, fmt.Errorf("ffprobe命令不可用: %v", err)
	}

	// 使用ffprobe获取音频时长
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filePath)

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		LogError("执行ffprobe命令失败: %v, stderr: %s", err, stderr.String())
		return 0, fmt.Errorf("执行ffprobe命令失败: %v", err)
	}

	// 解析输出结果
	durationStr := strings.TrimSpace(out.String())
	if durationStr == "" {
		LogWarning("ffprobe未返回有效的时长信息")
		return 0, fmt.Errorf("未获取到有效的时长信息")
	}

	duration, err := strconv.ParseFloat(durationStr, 64)
	if err != nil {
		LogError("解析音频时长失败: %v, 原始值: %s", err, durationStr)
		return 0, fmt.Errorf("解析音频时长失败: %v", err)
	}

	LogInfo("成功获取音频时长: %.2f秒", duration)
	return duration, nil
}

// GetAudioInfo 获取音频文件的完整信息
func GetAudioInfo(filePath string) (map[string]interface{}, error) {
	info := make(map[string]interface{})

	// 获取文件大小
	size, err := GetFileSize(filePath)
	if err == nil {
		info["size"] = size
	}

	// 获取音频时长
	duration, err := GetAudioDuration(filePath)
	if err == nil {
		info["duration"] = duration
	} else {
		info["duration"] = 0.0
	}

	// 获取文件类型
	info["type"] = strings.TrimPrefix(strings.ToLower(filepath.Ext(filePath)), ".")

	// 获取文件名
	info["name"] = filepath.Base(filePath)

	return info, nil
}
