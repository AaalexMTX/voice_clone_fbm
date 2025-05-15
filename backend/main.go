package main

import (
	"log"
	"github.com/gin-gonic/gin"
	"github.com/gin-contrib/cors"
)

func main() {
	// 创建Gin引擎
	r := gin.Default()

	// 配置CORS
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:3000"}
	r.Use(cors.New(config))

	// API路由组
	api := r.Group("/api")
	{
		// 音频处理相关接口
		api.POST("/upload", handleAudioUpload)
		api.POST("/clone", handleVoiceClone)
		api.GET("/status/:taskId", getTaskStatus)
	}

	// 启动服务器
	if err := r.Run(":8080"); err != nil {
		log.Fatal("Failed to start server: ", err)
	}
}

// 处理音频上传
func handleAudioUpload(c *gin.Context) {
	// TODO: 实现音频上传逻辑
	c.JSON(200, gin.H{
		"message": "Audio upload endpoint",
	})
}

// 处理语音克隆
func handleVoiceClone(c *gin.Context) {
	// TODO: 实现语音克隆逻辑
	c.JSON(200, gin.H{
		"message": "Voice clone endpoint",
	})
}

// 获取任务状态
func getTaskStatus(c *gin.Context) {
	taskId := c.Param("taskId")
	// TODO: 实现任务状态查询逻辑
	c.JSON(200, gin.H{
		"taskId": taskId,
		"status": "pending",
	})
}