package controller

import (
	"fmt"
	"net/http"

	"voice_clone_fbm/backend/model"

	"github.com/gin-gonic/gin"
)

// CheckDatabase 检查数据库结构
func CheckDatabase(c *gin.Context) {
	db := model.GetDB()
	if db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "数据库连接未初始化",
		})
		return
	}

	// 检查表是否存在
	var tables []string
	if err := db.Raw("SHOW TABLES").Pluck("Tables_in_"+db.Migrator().CurrentDatabase(), &tables).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("获取表列表失败: %v", err),
		})
		return
	}

	// 检查推理历史记录表是否存在
	inferenceHistoryExists := false
	for _, table := range tables {
		if table == "inference_histories" {
			inferenceHistoryExists = true
			break
		}
	}

	// 如果表不存在，尝试创建
	if !inferenceHistoryExists {
		if err := db.Migrator().CreateTable(&model.InferenceHistory{}); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": fmt.Sprintf("创建推理历史记录表失败: %v", err),
			})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"message": "推理历史记录表已创建",
			"tables":  append(tables, "inference_histories"),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "数据库检查完成",
		"tables":  tables,
	})
}
