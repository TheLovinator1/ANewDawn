package main

import (
	"context"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

/*
Prompt is the Discord message that the user sent to the bot.
*/
func GetGPT3Response(prompt string, author string) (string, error) {
	openAIToken := config.OpenAIToken
	if openAIToken == "" {
		return "", fmt.Errorf("OPENAI_API_KEY is not set")
	}

	// Remove the mention from the prompt
	prompt = strings.Replace(prompt, "@LoviBot ", "", -1)

	// Remove the mention from the prompt
	prompt = strings.Replace(prompt, "LoviBot ", "", -1)

	// Remove the mention from the prompt
	prompt = strings.Replace(prompt, "<@345000831499894795> ", "", -1)

	// Print the prompt
	fmt.Println("Prompt:", author, ":", prompt)

	// Check if the prompt is too long
	if len(prompt) > 2048 {
		return "", fmt.Errorf("prompt is too long")
	}

	// Check if the prompt is too short
	if len(prompt) < 1 {
		return "", fmt.Errorf("prompt is too short")
	}

	// Add additional information to the system message
	var additionalInfo string
	switch author {
	case "thelovinator":
		additionalInfo = "User (TheLovinator) is a programmer. Wants to live in the woods. Real name is Joakim. He made the bot."
	case "killyoy":
		additionalInfo = "User (KillYoy) likes to play video games. Real name is Andreas. Good at CSS."
	case "forgefilip":
		additionalInfo = "User (ForgeFilip) likes watches. Real name is Filip."
	case "plubplub":
		additionalInfo = "User (Piplup) likes to play WoW and Path of Exile. Real name is Axel. Is also called Bambi."
	case "nobot":
		additionalInfo = "User (Nobot) likes to play WoW. Real name is Gustav. Really good at programming."
	case "kao172":
		additionalInfo = "User (kao172) likes cars. Real name is Fredrik."
	}

	// Create a new client
	client := openai.NewClient(openAIToken)

	// System message
	var systemMessage string
	systemMessage = `You are in a Discord server. You are a space communist.
	You are Swedish. 
	Use Markdown for formatting. 
	Please respond with a short message.
	You want members of the server to like you and join Rymdkommunisterna AB.
	You want to be a good bot so you will answer the user's question.
	`

	// Add additional information to the system message
	if additionalInfo != "" {
		systemMessage = fmt.Sprintf("%s %s", systemMessage, additionalInfo)
	}

	// Print the system message
	fmt.Println("System message:", systemMessage)

	// Create a completion
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: systemMessage,
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		return "", fmt.Errorf("failed to get response from GPT-3: %v", err)
	}

	ourResponse := resp.Choices[0].Message.Content
	ourResponse = strings.Replace(ourResponse, "As a space communist, ", "", -1)

	fmt.Println("Response:", ourResponse)
	return ourResponse, nil
}
