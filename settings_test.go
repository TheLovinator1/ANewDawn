package main

import (
	"os"
	"reflect"
	"testing"
)

// Returns a Config object with DiscordToken set when DISCORD_TOKEN environment variable is set
func TestLoadFromEnvironment_DiscordTokenSet(t *testing.T) {
	os.Setenv("DISCORD_TOKEN", "test_token")
	defer os.Unsetenv("DISCORD_TOKEN")

	config, err := loadFromEnvironment()
	if err != nil {
		t.Errorf("Expected no error, but got: %v", err)
	}

	expected := &Config{
		DiscordToken: "test_token",
	}
	if !reflect.DeepEqual(config, expected) {
		t.Errorf("Expected config to be %v, but got %v", expected, config)
	}
}

// Returns an error when DISCORD_TOKEN environment variable is empty
func TestLoadFromEnvironment_EmptyDiscordToken(t *testing.T) {
	os.Setenv("DISCORD_TOKEN", "")
	defer os.Unsetenv("DISCORD_TOKEN")

	_, err := loadFromEnvironment()
	if err == nil {
		t.Error("Expected an error, but got nil")
	}

	expected := "DISCORD_TOKEN environment variable not set or empty. Also tried reading from settings.json file"
	if err.Error() != expected {
		t.Errorf("Expected error message to be '%s', but got '%s'", expected, err.Error())
	}
}

// Returns an error when DISCORD_TOKEN environment variable is not set and settings.json file is not present
func TestLoadFromEnvironment_NoDiscordTokenNoSettingsFile(t *testing.T) {
	os.Unsetenv("DISCORD_TOKEN")

	_, err := loadFromEnvironment()
	if err == nil {
		t.Error("Expected an error, but got nil")
	}

	expected := "DISCORD_TOKEN environment variable not set or empty. Also tried reading from settings.json file"
	if err.Error() != expected {
		t.Errorf("Expected error message to be '%s', but got '%s'", expected, err.Error())
	}
}

// Returns an error when settings.json file is present but DiscordToken is not set
func TestLoadFromEnvironment_SettingsFileNoDiscordToken(t *testing.T) {
	os.Setenv("DISCORD_TOKEN", "")
	defer os.Unsetenv("DISCORD_TOKEN")

	_, err := loadFromEnvironment()
	if err == nil {
		t.Error("Expected an error, but got nil")
	}

	expected := "DISCORD_TOKEN environment variable not set or empty. Also tried reading from settings.json file"
	if err.Error() != expected {
		t.Errorf("Expected error message to be '%s', but got '%s'", expected, err.Error())
	}
}
