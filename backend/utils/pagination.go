package utils

import (
	"strconv"

	"github.com/gin-gonic/gin"
)

// 默认分页参数
const (
	DefaultPage     = 1
	DefaultPageSize = 10
	MaxPageSize     = 100
)

// GetPagination 从请求中获取分页参数
func GetPagination(c *gin.Context) (int, int) {
	// 获取页码
	pageStr := c.DefaultQuery("page", "1")
	page, err := strconv.Atoi(pageStr)
	if err != nil || page < 1 {
		page = DefaultPage
	}

	// 获取每页数量
	pageSizeStr := c.DefaultQuery("pageSize", "10")
	pageSize, err := strconv.Atoi(pageSizeStr)
	if err != nil || pageSize < 1 {
		pageSize = DefaultPageSize
	}

	// 限制最大每页数量
	if pageSize > MaxPageSize {
		pageSize = MaxPageSize
	}

	return page, pageSize
}
