package controller

import (
	"net/http"
	"time"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// SaveInferenceHistory 保存推理历史记录
func SaveInferenceHistory(c *gin.Context) {
	// 获取当前用户ID
	uid, exists := c.Get("uid")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未授权"})
		return
	}

	// 解析请求参数
	var req struct {
		Name     string  `json:"name"`     // 音频名称
		Text     string  `json:"text"`     // 输入文本
		ModelId  string  `json:"modelId"`  // 模型ID
		AudioUrl string  `json:"audioUrl"` // 音频URL
		Speed    float32 `json:"speed"`    // 语速
		Pitch    float32 `json:"pitch"`    // 音调
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
		return
	}

	// 验证必填字段
	if req.Text == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "输入文本不能为空"})
		return
	}

	// 如果没有提供名称，使用默认名称
	if req.Name == "" {
		req.Name = "合成音频_" + time.Now().Format("20060102150405")
	}

	// 处理模型ID
	modelId := req.ModelId
	if modelId == "default" {
		modelId = "system_default_model" // 系统默认模型ID
	}

	// 处理音频URL
	audioPath := req.AudioUrl
	if audioPath == "" {
		audioPath = "assets/default_audio.mp3" // 默认音频路径
	}

	// 创建历史记录
	history := model.InferenceHistory{
		HID:        uuid.New().String()[:32], // 生成唯一ID
		UID:        uid.(string),
		MID:        modelId,
		InputText:  req.Text,
		OutputPath: audioPath,
		AudioName:  req.Name,
		Status:     2, // 已完成
		Duration:   0, // 暂时不计算时长
	}

	// 保存到数据库
	db := model.GetDB()
	if err := history.Create(db); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 0,
		"msg":  "保存成功",
		"data": gin.H{
			"hid":        history.HID,
			"created_at": history.CreatedAt,
		},
	})
}

// GetUserInferenceHistories 获取用户的推理历史记录
func GetUserInferenceHistories(c *gin.Context) {
	// 获取当前用户ID
	uid, exists := c.Get("uid")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未授权"})
		return
	}

	// 获取分页参数
	page, pageSize := utils.GetPagination(c)

	// 获取历史记录
	var history model.InferenceHistory
	db := model.GetDB()

	// 获取详细信息
	details, err := history.GetHistoryDetailsByUID(db, uid.(string))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取历史记录失败"})
		return
	}

	// 分页处理
	start := (page - 1) * pageSize
	end := start + pageSize
	if start >= len(details) {
		details = []model.HistoryDetail{}
	} else if end > len(details) {
		details = details[start:]
	} else {
		details = details[start:end]
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 0,
		"msg":  "获取成功",
		"data": gin.H{
			"histories": details,
			"total":     len(details),
			"page":      page,
			"pageSize":  pageSize,
		},
	})
}

// GetInferenceHistoryDetail 获取推理历史记录详情
func GetInferenceHistoryDetail(c *gin.Context) {
	// 获取历史记录ID
	hid := c.Param("hid")
	if hid == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
		return
	}

	// 获取历史记录详情
	var history model.InferenceHistory
	db := model.GetDB()
	detail, err := history.GetHistoryDetailByHID(db, hid)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取历史记录失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 0,
		"msg":  "获取成功",
		"data": detail,
	})
}

// DeleteInferenceHistory 删除推理历史记录
func DeleteInferenceHistory(c *gin.Context) {
	// 获取当前用户ID
	uid, exists := c.Get("uid")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未授权"})
		return
	}

	// 获取历史记录ID
	hid := c.Param("hid")
	if hid == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
		return
	}

	// 删除历史记录
	var history model.InferenceHistory
	db := model.GetDB()

	// 先检查历史记录是否存在且属于当前用户
	exists, err := history.CheckHistoryExists(db, hid, uid.(string))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "查询历史记录失败"})
		return
	}

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "历史记录不存在或无权限删除"})
		return
	}

	// 执行删除
	if err := history.DeleteByHID(db, hid); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 0,
		"msg":  "删除成功",
	})
}
