package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/utils"

	"github.com/google/uuid"
	log "github.com/sirupsen/logrus"
)

// AudioUploadReq 音频上传请求
type AudioUploadReq struct {
	File     *multipart.FileHeader
	Username string
}

// AudioUploadResp 音频上传响应
type AudioUploadResp struct {
	AudioID   string
	Name      string
	Status    int8
	FileType  string
	FilePath  string
	Duration  float64
	CreatedAt time.Time
}

// UploadAudio 上传音频文件
func UploadAudio(req *AudioUploadReq) (*AudioUploadResp, error) {
	// 检查文件类型
	ext := strings.ToLower(filepath.Ext(req.File.Filename))
	if ext != ".wav" && ext != ".mp3" {
		log.Errorf("用户 %s 尝试上传不支持的文件类型: %s", req.Username, ext)
		return nil, fmt.Errorf("不支持的文件类型，仅支持WAV和MP3")
	}

	// 获取用户信息，包括UID
	user, err := model.GetByUsername(req.Username)
	if err != nil {
		log.Errorf("获取用户信息失败: %v", err)
		return nil, fmt.Errorf("获取用户信息失败: %v", err)
	}

	// 确保用户目录存在
	if err := utils.EnsureUserDir(req.Username); err != nil {
		log.Errorf("创建用户目录失败: %v", err)
		return nil, fmt.Errorf("创建用户目录失败: %v", err)
	}

	// 获取用户音频保存路径
	sourceDir := filepath.Join(utils.DataDir, req.Username, "source")
	log.Infof("用户音频目录: %s", sourceDir)

	// 再次确保source目录存在
	if err := utils.CreateDirIfNotExist(sourceDir); err != nil {
		log.Errorf("创建用户音频目录失败: %v", err)
		return nil, fmt.Errorf("创建用户音频目录失败: %v", err)
	}

	// 使用原始文件名
	filePath := utils.GetUserVoicePath(req.Username, req.File.Filename)

	// 创建目标文件
	dst, err := os.Create(filePath)
	if err != nil {
		log.Errorf("创建目标文件失败: %v", err)
		return nil, fmt.Errorf("创建目标文件失败: %v", err)
	}
	defer dst.Close()

	// 打开源文件
	src, err := req.File.Open()
	if err != nil {
		log.Errorf("打开上传文件失败: %v", err)
		return nil, fmt.Errorf("打开上传文件失败: %v", err)
	}
	defer src.Close()

	// 复制文件内容
	fileSize, err := io.Copy(dst, src)
	if err != nil {
		log.Errorf("复制文件内容失败: %v", err)
		return nil, fmt.Errorf("复制文件内容失败: %v", err)
	}

	log.Infof("成功保存文件 %s，大小: %d 字节", filePath, fileSize)

	// 检查文件是否成功保存
	if !utils.FileExists(filePath) {
		log.Errorf("文件保存失败，无法找到文件: %s", filePath)
		return nil, fmt.Errorf("文件保存失败，无法找到文件: %s", filePath)
	}

	// 获取音频时长
	duration := 0.0
	audioInfo, err := utils.GetAudioDuration(filePath)
	if err != nil {
		log.Warnf("获取音频时长失败: %v，将使用默认值0", err)
	} else {
		duration = audioInfo
		log.Infof("获取到音频时长: %.2f秒", duration)
	}

	log.Infof("文件保存成功，即将创建数据库记录，音频时长: %.2f秒", duration)

	// 创建音频记录，AID将在BeforeCreate中自动生成
	audio := &model.Audio{
		Name:     req.File.Filename,
		FilePath: filePath,
		FileType: strings.TrimPrefix(ext, "."),
		Duration: duration,
		Status:   model.AudioProcessPending,
	}
	if err := audio.Create(model.DB); err != nil {
		log.Errorf("创建音频记录失败: %v", err)
		return nil, fmt.Errorf("创建音频记录失败: %v", err)
	}

	// 确保AID已生成
	if audio.AID == "" {
		log.Errorf("音频AID生成失败")
		return nil, fmt.Errorf("音频AID生成失败")
	}

	// 创建用户音频关联
	userAudio := &model.UserAudio{
		UID:  user.UID,   // 使用用户的UID而不是用户名
		AID:  audio.AID,  // 使用自动生成的AID
		Name: audio.Name, // 添加音频名称
	}
	if err := userAudio.Create(model.DB); err != nil {
		log.Errorf("创建用户音频关联失败: %v", err)
		return nil, fmt.Errorf("创建用户音频关联失败: %v", err)
	}

	log.Infof("音频上传成功: ID=%d, Path=%s, AID=%s, UID=%s, 用户=%s, 文件名=%s, 时长=%.2f秒",
		audio.ID, audio.FilePath, audio.AID, user.UID, req.Username, req.File.Filename, duration)

	// 返回响应
	return &AudioUploadResp{
		AudioID:   audio.AID,
		Name:      audio.Name,
		Status:    audio.Status,
		FileType:  audio.FileType,
		FilePath:  audio.FilePath,
		Duration:  audio.Duration,
		CreatedAt: time.Now(),
	}, nil
}

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

// GetUserAudios 获取用户的音频列表
func GetUserAudios(username string) ([]model.Audio, error) {
	log.Infof("获取用户 %s 的音频列表", username)

	// 1. 根据用户名获取用户信息
	user, err := model.GetByUsername(username)
	if err != nil {
		log.Errorf("获取用户信息失败: %v", err)
		return nil, fmt.Errorf("获取用户信息失败: %v", err)
	}

	// 2. 根据用户UID查询音频列表
	audios, err := model.GetUserAudios(user.UID)
	if err != nil {
		log.Errorf("获取音频列表失败: %v", err)
		return nil, fmt.Errorf("获取音频列表失败: %v", err)
	}

	log.Infof("成功获取用户 %s 的音频列表，共 %d 条记录", username, len(audios))
	return audios, nil
}

// DeleteAudio 删除音频
func DeleteAudio(username string, audioID string) error {
	log.Infof("用户 %s 请求删除音频 ID: %s", username, audioID)

	// 1. 根据用户名获取用户信息
	user, err := model.GetByUsername(username)
	if err != nil {
		log.Errorf("获取用户信息失败: %v", err)
		return fmt.Errorf("获取用户信息失败: %v", err)
	}

	// 2. 获取音频记录
	var audio model.Audio
	if err := audio.GetByAID(audioID); err != nil {
		log.Errorf("音频不存在: %v", err)
		return fmt.Errorf("音频不存在")
	}

	// 3. 检查用户权限
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND a_id = ?", user.UID, audioID).First(&userAudio).Error; err != nil {
		log.Warnf("用户无权访问该音频: %v", err)
		return fmt.Errorf("无权访问该音频")
	}

	// 记录文件路径，用于日志
	filePath := audio.FilePath
	fileName := audio.Name

	// 4. 删除音频文件和记录
	if err := audio.Delete(); err != nil {
		log.Errorf("删除音频失败: %v", err)
		return fmt.Errorf("删除音频失败: %v", err)
	}

	// 记录详细日志
	log.Infof("成功删除音频: ID=%s, 用户=%s, 文件名=%s, 路径=%s", audioID, username, fileName, filePath)
	return nil
}

// UpdateAudioContent 更新音频内容
func UpdateAudioContent(audioID, content string, username string) error {
	log.Infof("用户 %s 请求更新音频 ID: %s 的内容", username, audioID)

	// 1. 根据用户名获取用户信息
	user, err := model.GetByUsername(username)
	if err != nil {
		log.Errorf("获取用户信息失败: %v", err)
		return fmt.Errorf("获取用户信息失败: %v", err)
	}

	// 2. 获取音频记录
	var audio model.Audio
	if err := audio.GetByAID(audioID); err != nil {
		log.Errorf("音频不存在: %v", err)
		return fmt.Errorf("音频不存在")
	}

	// 3. 检查用户权限
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND a_id = ?", user.UID, audioID).First(&userAudio).Error; err != nil {
		log.Warnf("用户无权访问该音频: %v", err)
		return fmt.Errorf("无权访问该音频")
	}

	// 4. 更新音频内容
	if err := audio.UpdateContent(content); err != nil {
		log.Errorf("更新音频内容失败: %v", err)
		return fmt.Errorf("更新音频内容失败: %v", err)
	}

	log.Infof("成功更新音频内容: ID=%s, 用户=%s", audioID, username)
	return nil
}
