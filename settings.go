package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the configuration parameters
type Config struct {
	DiscordToken string `json:"discord_token"`
	OpenAIToken  string `json:"openai_token"`
}

// Load reads configuration from settings.json or environment variables
func Load() (*Config, error) {
	// Try reading from settings.json file first
	config, err := loadFromJSONFile("settings.json")
	if err != nil {
		// If reading from file fails, try reading from environment variables
		config, err = loadFromEnvironment()
		if err != nil {
			return nil, err
		}
	}

	return config, nil
}

// loadFromJSONFile reads configuration from a JSON file
func loadFromJSONFile(filename string) (*Config, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open settings file: %v", err)
	}
	defer file.Close()

	config := &Config{}
	decoder := json.NewDecoder(file)
	err = decoder.Decode(config)
	if err != nil {
		return nil, fmt.Errorf("failed to decode settings file: %v", err)
	}

	return config, nil
}

// loadFromEnvironment reads configuration from environment variables
func loadFromEnvironment() (*Config, error) {
	discordToken := os.Getenv("DISCORD_TOKEN")
	if discordToken == "" {
		return nil, fmt.Errorf("DISCORD_TOKEN environment variable not set or empty. Also tried reading from settings.json file")
	}

	openAIToken := os.Getenv("OPENAI_TOKEN")
	if openAIToken == "" {
		return nil, fmt.Errorf("OPENAI_TOKEN environment variable not set or empty. Also tried reading from settings.json file")
	}

	config := &Config{
		DiscordToken: discordToken,
		OpenAIToken:  openAIToken,
	}

	return config, nil
}
