package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

// AudioService 处理音频相关的业务逻辑
type AudioService struct {
	UploadDir       string
	ModelServiceURL string
}

// NewAudioService 创建新的AudioService实例
func NewAudioService() *AudioService {
	return &AudioService{
		UploadDir:       "uploads",
		ModelServiceURL: "http://localhost:5000",
	}
}

// SaveAudio 保存上传的音频文件
func (s *AudioService) SaveAudio(data []byte, filename string) (string, error) {
	// 生成唯一的文件名
	fileID := uuid.New().String()
	ext := filepath.Ext(filename)
	newFilename := fmt.Sprintf("%s%s", fileID, ext)

	// 确保上传目录存在
	if err := os.MkdirAll(s.UploadDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create upload directory: %v", err)
	}

	// 保存文件
	filePath := filepath.Join(s.UploadDir, newFilename)
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return "", fmt.Errorf("failed to save file: %v", err)
	}

	return fileID, nil
}

// Task 表示语音克隆任务
type Task struct {
	ID        string    `json:"id"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	AudioID   string    `json:"audio_id"`
}

// CloneVoiceRequest 表示发送给模型服务的请求
type CloneVoiceRequest struct {
	TaskID  string `json:"task_id"`
	AudioID string `json:"audio_id"`
	Text    string `json:"text,omitempty"`
}

// CloneVoiceResponse 表示从模型服务收到的响应
type CloneVoiceResponse struct {
	Message string `json:"message"`
	TaskID  string `json:"task_id"`
	Error   string `json:"error,omitempty"`
}

// CloneVoice 处理语音克隆请求
func (s *AudioService) CloneVoice(audioID string) (*Task, error) {
	// 创建新任务
	task := &Task{
		ID:        uuid.New().String(),
		Status:    "processing",
		CreatedAt: time.Now(),
		AudioID:   audioID,
	}

	// 准备API请求
	reqBody := CloneVoiceRequest{
		TaskID:  task.ID,
		AudioID: audioID,
	}

	reqData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// 发送请求到Python模型服务
	resp, err := http.Post(
		fmt.Sprintf("%s/api/clone", s.ModelServiceURL),
		"application/json",
		bytes.NewBuffer(reqData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to model service: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model service returned error status: %d", resp.StatusCode)
	}

	// 解析响应
	var apiResp CloneVoiceResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	// 检查错误
	if apiResp.Error != "" {
		return nil, fmt.Errorf("model service error: %s", apiResp.Error)
	}

	return task, nil
}

// TaskStatusResponse 表示任务状态的API响应
type TaskStatusResponse struct {
	TaskID     string  `json:"task_id"`
	Status     string  `json:"status"`
	CreatedAt  float64 `json:"created_at"`
	ResultPath string  `json:"result_path,omitempty"`
	Error      string  `json:"error,omitempty"`
}

// GetTaskStatus 获取任务状态
func (s *AudioService) GetTaskStatus(taskID string) (*Task, error) {
	// 发送请求到Python模型服务
	resp, err := http.Get(fmt.Sprintf("%s/api/task/%s", s.ModelServiceURL, taskID))
	if err != nil {
		return nil, fmt.Errorf("failed to get task status: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model service returned error status: %d", resp.StatusCode)
	}

	// 解析响应
	var statusResp TaskStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&statusResp); err != nil {
		return nil, fmt.Errorf("failed to decode status response: %v", err)
	}

	// 转换为Task结构
	task := &Task{
		ID:        statusResp.TaskID,
		Status:    statusResp.Status,
		CreatedAt: time.Unix(int64(statusResp.CreatedAt), 0),
	}

	return task, nil
}
