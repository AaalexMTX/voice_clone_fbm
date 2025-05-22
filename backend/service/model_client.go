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
	"time"

	"voice_clone_fbm/backend/config"

	log "github.com/sirupsen/logrus"
)

// ModelClient 模型API客户端
type ModelClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewModelClient 创建一个新的模型API客户端
func NewModelClient() *ModelClient {
	// 从配置中读取模型服务URL
	baseURL := config.GlobalConfig.Model.APIBaseURL
	if baseURL == "" {
		baseURL = "http://localhost:7860/api" // 默认模型服务地址
	}

	return &ModelClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// UploadReferenceAudio 上传参考音频
func (c *ModelClient) UploadReferenceAudio(audioPath string) (string, error) {
	url := fmt.Sprintf("%s/upload_reference", c.BaseURL)

	// 打开文件
	file, err := os.Open(audioPath)
	if err != nil {
		log.Errorf("打开音频文件失败: %v", err)
		return "", fmt.Errorf("打开音频文件失败: %v", err)
	}
	defer file.Close()

	// 创建multipart表单
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("audio", filepath.Base(audioPath))
	if err != nil {
		log.Errorf("创建表单字段失败: %v", err)
		return "", fmt.Errorf("创建表单字段失败: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		log.Errorf("复制文件内容失败: %v", err)
		return "", fmt.Errorf("复制文件内容失败: %v", err)
	}

	// 关闭writer
	err = writer.Close()
	if err != nil {
		log.Errorf("关闭writer失败: %v", err)
		return "", fmt.Errorf("关闭writer失败: %v", err)
	}

	// 创建请求
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		log.Errorf("创建HTTP请求失败: %v", err)
		return "", fmt.Errorf("创建HTTP请求失败: %v", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// 发送请求
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		log.Errorf("发送HTTP请求失败: %v", err)
		return "", fmt.Errorf("发送HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 解析响应
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Errorf("服务器返回错误状态码: %d, 响应: %s", resp.StatusCode, string(bodyBytes))
		return "", fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	var result struct {
		Status           string `json:"status"`
		ReferenceAudioID string `json:"reference_audio_id"`
		ReferencePath    string `json:"reference_path"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Errorf("解析响应失败: %v", err)
		return "", fmt.Errorf("解析响应失败: %v", err)
	}

	if result.Status != "success" {
		log.Errorf("上传参考音频失败: %s", result.Status)
		return "", fmt.Errorf("上传参考音频失败")
	}

	return result.ReferenceAudioID, nil
}

// ExtractEmbedding 从音频提取说话人嵌入
func (c *ModelClient) ExtractEmbedding(audioPath string) (string, error) {
	url := fmt.Sprintf("%s/extract_embedding", c.BaseURL)

	// 打开文件
	file, err := os.Open(audioPath)
	if err != nil {
		log.Errorf("打开音频文件失败: %v", err)
		return "", fmt.Errorf("打开音频文件失败: %v", err)
	}
	defer file.Close()

	// 创建multipart表单
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("audio", filepath.Base(audioPath))
	if err != nil {
		log.Errorf("创建表单字段失败: %v", err)
		return "", fmt.Errorf("创建表单字段失败: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		log.Errorf("复制文件内容失败: %v", err)
		return "", fmt.Errorf("复制文件内容失败: %v", err)
	}

	// 关闭writer
	err = writer.Close()
	if err != nil {
		log.Errorf("关闭writer失败: %v", err)
		return "", fmt.Errorf("关闭writer失败: %v", err)
	}

	// 创建请求
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		log.Errorf("创建HTTP请求失败: %v", err)
		return "", fmt.Errorf("创建HTTP请求失败: %v", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// 发送请求
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		log.Errorf("发送HTTP请求失败: %v", err)
		return "", fmt.Errorf("发送HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 解析响应
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Errorf("服务器返回错误状态码: %d, 响应: %s", resp.StatusCode, string(bodyBytes))
		return "", fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	var result struct {
		Status        string `json:"status"`
		EmbeddingID   string `json:"embedding_id"`
		EmbeddingPath string `json:"embedding_path"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Errorf("解析响应失败: %v", err)
		return "", fmt.Errorf("解析响应失败: %v", err)
	}

	if result.Status != "success" {
		log.Errorf("提取嵌入失败: %s", result.Status)
		return "", fmt.Errorf("提取嵌入失败")
	}

	return result.EmbeddingID, nil
}

// SynthesizeSpeech 合成语音
func (c *ModelClient) SynthesizeSpeech(text string, embeddingID string) (string, error) {
	url := fmt.Sprintf("%s/synthesize", c.BaseURL)

	// 创建请求体
	reqBody := map[string]interface{}{
		"text":         text,
		"embedding_id": embeddingID,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		log.Errorf("编码JSON数据失败: %v", err)
		return "", fmt.Errorf("编码JSON数据失败: %v", err)
	}

	// 创建请求
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Errorf("创建HTTP请求失败: %v", err)
		return "", fmt.Errorf("创建HTTP请求失败: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// 发送请求
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		log.Errorf("发送HTTP请求失败: %v", err)
		return "", fmt.Errorf("发送HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 解析响应
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Errorf("服务器返回错误状态码: %d, 响应: %s", resp.StatusCode, string(bodyBytes))
		return "", fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	var result struct {
		Status     string `json:"status"`
		OutputID   string `json:"output_id"`
		OutputPath string `json:"output_path"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Errorf("解析响应失败: %v", err)
		return "", fmt.Errorf("解析响应失败: %v", err)
	}

	if result.Status != "success" {
		log.Errorf("合成语音失败: %s", result.Status)
		return "", fmt.Errorf("合成语音失败")
	}

	return result.OutputID, nil
}

// CloneVoice 一站式语音克隆
func (c *ModelClient) CloneVoice(text string, audioPath string) (string, error) {
	url := fmt.Sprintf("%s/tts", c.BaseURL)

	// 打开文件
	file, err := os.Open(audioPath)
	if err != nil {
		log.Errorf("打开音频文件失败: %v", err)
		return "", fmt.Errorf("打开音频文件失败: %v", err)
	}
	defer file.Close()

	// 创建multipart表单
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// 添加文本字段
	if err := writer.WriteField("text", text); err != nil {
		log.Errorf("添加文本字段失败: %v", err)
		return "", fmt.Errorf("添加文本字段失败: %v", err)
	}

	// 添加音频文件
	part, err := writer.CreateFormFile("audio", filepath.Base(audioPath))
	if err != nil {
		log.Errorf("创建表单字段失败: %v", err)
		return "", fmt.Errorf("创建表单字段失败: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		log.Errorf("复制文件内容失败: %v", err)
		return "", fmt.Errorf("复制文件内容失败: %v", err)
	}

	// 关闭writer
	err = writer.Close()
	if err != nil {
		log.Errorf("关闭writer失败: %v", err)
		return "", fmt.Errorf("关闭writer失败: %v", err)
	}

	// 创建请求
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		log.Errorf("创建HTTP请求失败: %v", err)
		return "", fmt.Errorf("创建HTTP请求失败: %v", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// 发送请求
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		log.Errorf("发送HTTP请求失败: %v", err)
		return "", fmt.Errorf("发送HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 解析响应
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Errorf("服务器返回错误状态码: %d, 响应: %s", resp.StatusCode, string(bodyBytes))
		return "", fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	var result struct {
		Status    string `json:"status"`
		OutputID  string `json:"output_id"`
		OutputURL string `json:"output_url"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Errorf("解析响应失败: %v", err)
		return "", fmt.Errorf("解析响应失败: %v", err)
	}

	if result.Status != "success" {
		log.Errorf("语音克隆失败: %s", result.Status)
		return "", fmt.Errorf("语音克隆失败")
	}

	return result.OutputID, nil
}

// DownloadAudio 下载生成的音频
func (c *ModelClient) DownloadAudio(outputID, savePath string) error {
	url := fmt.Sprintf("%s/audio/%s", c.BaseURL, outputID)

	// 创建请求
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		log.Errorf("创建HTTP请求失败: %v", err)
		return fmt.Errorf("创建HTTP请求失败: %v", err)
	}

	// 发送请求
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		log.Errorf("发送HTTP请求失败: %v", err)
		return fmt.Errorf("发送HTTP请求失败: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Errorf("服务器返回错误状态码: %d, 响应: %s", resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("服务器返回错误状态码: %d", resp.StatusCode)
	}

	// 创建输出文件
	out, err := os.Create(savePath)
	if err != nil {
		log.Errorf("创建输出文件失败: %v", err)
		return fmt.Errorf("创建输出文件失败: %v", err)
	}
	defer out.Close()

	// 将响应体内容复制到文件
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		log.Errorf("写入文件失败: %v", err)
		return fmt.Errorf("写入文件失败: %v", err)
	}

	log.Infof("音频已下载到: %s", savePath)
	return nil
}
