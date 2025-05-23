package router

import (
	"voice_clone_fbm/backend/controller"

	"voice_clone_fbm/backend/middleware"

	"github.com/gin-gonic/gin"
)

func InitRouter() *gin.Engine {
	r := gin.Default()

	// 允许跨域
	r.Use(middleware.Cors())

	// API路由组
	api := r.Group("/api")
	{
		// 认证相关路由（无需认证）
		api.POST("/register", controller.Register)
		api.POST("/login", controller.Login)

		// 数据库检查路由（调试用）
		api.GET("/db/check", controller.CheckDatabase)

		// 需要认证的路由组
		authenticated := api.Group("")
		authenticated.Use(middleware.JWTAuth())
		{
			// 用户相关
			user := authenticated.Group("/user")
			{
				user.PUT("/info", controller.UpdateUserInfo)
			}

			// 音频相关
			audio := authenticated.Group("/audio")
			{
				audio.POST("/upload", controller.UploadAudio)
				audio.GET("/list", controller.GetUserAudios)
				audio.DELETE("/:id", controller.DeleteAudio)
				audio.PUT("/:id/content", controller.UpdateAudioContent)
				audio.GET("/stream/:id", controller.StreamAudio)
			}

			// 模型训练相关
			model := authenticated.Group("/model")
			{
				// 开始训练模型
				model.POST("/train", controller.StartTraining)
				// 获取用户的所有模型
				model.GET("/list", controller.GetUserModels)
			}

			// 推理历史记录相关
			inference := authenticated.Group("/inference")
			{
				// 保存推理历史记录
				inference.POST("/save", controller.SaveInferenceHistory)
				// 获取用户的推理历史记录
				inference.GET("/list", controller.GetUserInferenceHistories)
				// 获取推理历史记录详情
				inference.GET("/detail/:hid", controller.GetInferenceHistoryDetail)
			}
		}
	}

	return r
}
