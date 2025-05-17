package middleware

import (
	"strings"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt"
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
		}

		c.Next()
	}
}
