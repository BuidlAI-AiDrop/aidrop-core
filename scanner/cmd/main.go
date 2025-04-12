package main

import (
	"flag"

	"github.com/rrabit42/aidrop-core/internal/config"
	"github.com/rrabit42/aidrop-core/internal/service"
)

var configFlag = flag.String("config", "./config.toml", "configuration toml file path")

func main() {
	cfg := config.NewConfig(*configFlag)
	service.NewService(cfg)
}
