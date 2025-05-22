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
				// audio.GET("/stream/:id", controller.StreamAudio)
			}
		}
	}

	return r
}
