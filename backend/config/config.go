package config

import (
	"io/ioutil"

	"gopkg.in/yaml.v2"
)

type Config struct {
	Server struct {
		Port int    `yaml:"port"`
		Host string `yaml:"host"`
	} `yaml:"server"`

	Database struct {
		Host      string `yaml:"host"`
		Port      int    `yaml:"port"`
		Username  string `yaml:"username"`
		Password  string `yaml:"password"`
		DBName    string `yaml:"dbname"`
		Charset   string `yaml:"charset"`
		ParseTime bool   `yaml:"parse_time"`
		Loc       string `yaml:"loc"`
	} `yaml:"database"`
}

var GlobalConfig Config

func Init() error {
	data, err := ioutil.ReadFile("config/config.yaml")
	if err != nil {
		return err
	}

	err = yaml.Unmarshal(data, &GlobalConfig)
	if err != nil {
		return err
	}

	return nil
}
