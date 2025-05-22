package middleware

import (
	"strings"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt"
	log "github.com/sirupsen/logrus"
)

func JWTAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		auth := c.GetHeader("Authorization")
		if auth == "" {
			c.JSON(401, gin.H{"error": "未提供认证信息"})
			c.Abort()
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		claims := jwt.MapClaims{}

		_, err := jwt.ParseWithClaims(token, claims, func(token *jwt.Token) (interface{}, error) {
			return []byte(utils.SecretKey), nil
		})

		if err != nil {
			c.JSON(401, gin.H{"error": "无效的token"})
			c.Abort()
			return
		}

		if username, ok := claims["username"].(string); ok {
			c.Set("username", username)

			// 从数据库获取用户UID
			user, err := model.GetByUsername(username)
			if err != nil {
				log.Errorf("无法获取用户信息: %v", err)
				c.JSON(500, gin.H{"error": "系统错误"})
				c.Abort()
				return
			}

			// 设置UID到上下文
			c.Set("uid", user.UID)
		}

		c.Next()
	}
}
