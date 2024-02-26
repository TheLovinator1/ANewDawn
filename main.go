package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"strings"

	"github.com/bwmarrin/discordgo"

	"github.com/vartanbeno/go-reddit/v2/reddit"
)

var config Config

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

func init() {
	loadedConfig, err := Load()
	if err != nil {
		log.Fatal(err)
	}
	config = *loadedConfig
}

func GetPostsFromReddit(subreddit string) (string, error) {
	if subreddit == "" {
		return "", fmt.Errorf("subreddit cannot be empty")
	}

	client, err := reddit.NewReadonlyClient()
	if err != nil {
		log.Println("Failed to create Reddit client:", err)
		return "", err
	}

	posts, _, err := client.Subreddit.TopPosts(context.Background(), subreddit, &reddit.ListPostOptions{
		ListOptions: reddit.ListOptions{
			Limit: 100,
		},
		Time: "all",
	})
	if err != nil {
		return "", fmt.Errorf("failed to get posts from Reddit: %v", err)
	}

	// Check if the subreddit exists
	if len(posts) == 0 {
		return "", fmt.Errorf("subreddit '%v' does not exist", subreddit)
	}

	// [Title](<https://old.reddit.com{Permalink}>)\n{URL}
	randInt := rand.Intn(len(posts))
	discordMessage := fmt.Sprintf("[%v](<https://old.reddit.com%v>)\n%v", posts[randInt].Title, posts[randInt].Permalink, posts[randInt].URL)

	return discordMessage, nil

}

func handleRedditCommand(s *discordgo.Session, i *discordgo.InteractionCreate, subreddit string) {
	post, err := GetPostsFromReddit(subreddit)
	if err = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: fmt.Sprintf("Cannot get a random post: %v", err),
			Flags:   discordgo.MessageFlagsEphemeral,
		},
	}); err != nil {
		log.Println("Failed to respond to interaction:", err)
		return
	}

	if err := s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: post,
		},
	}); err != nil {
		log.Println("Failed to respond to interaction:", err)
		return
	}
}

var (
	commands = []*discordgo.ApplicationCommand{
		{
			Name:        "dank_memes",
			Description: "Sends dank meme from /r/GoodAnimemes",
		},
		{
			Name:        "waifus",
			Description: "Sends waifu from /r/WatchItForThePlot",
		},
		{
			Name:        "milkers",
			Description: "Sends milkers from /r/RetrousseTits",
		},
		{
			Name:        "thighs",
			Description: "Sends thighs from /r/ZettaiRyouiki",
		},
	}

	commandHandlers = map[string]func(s *discordgo.Session, i *discordgo.InteractionCreate){
		// Dank memes command
		"dank_memes": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			handleRedditCommand(s, i, "GoodAnimemes")
		},

		// Waifus command
		"waifus": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			handleRedditCommand(s, i, "WatchItForThePlot")
		},

		// Milkers command
		"milkers": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			handleRedditCommand(s, i, "RetrousseTits")
		},

		// Thighs command
		"thighs": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			handleRedditCommand(s, i, "ZettaiRyouiki")
		},
		// Echo command
		"echo": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			// Check if the user provided a message
			if len(i.ApplicationCommandData().Options) == 0 {
				// If not, send an ephemeral message to the user
				err := s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
					Type: discordgo.InteractionResponseChannelMessageWithSource,
					Data: &discordgo.InteractionResponseData{
						Content: "You need to provide a message!",
						Flags:   discordgo.MessageFlagsEphemeral,
					},
				})
				if err != nil {
					log.Println("Failed to respond to interaction:", err)
					return
				}
			}

			// Check that the option is not empty
			if i.ApplicationCommandData().Options[0].StringValue() == "" {
				// If not, send an ephemeral message to the user
				err := s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
					Type: discordgo.InteractionResponseChannelMessageWithSource,
					Data: &discordgo.InteractionResponseData{
						Content: "The message cannot be empty!",
						Flags:   discordgo.MessageFlagsEphemeral,
					},
				})
				if err != nil {
					log.Println("Failed to respond to interaction:", err)
					return
				}
			}

			// Respond to the original message so we don't get "This interaction failed" error
			if _, err := s.ChannelMessageSend(i.ChannelID, i.ApplicationCommandData().Options[0].StringValue()); err != nil {
				log.Println("Failed to send message to channel:", err)
				return
			}
		},
	}
)

func onMessageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	if m.Author.ID == s.State.User.ID {
		return
	}

	allowedUsers := []string{
		"thelovinator",
		"killyoy",
		"forgefilip",
		"plubplub",
		"nobot",
		"kao172",
	}

	// Have a 1/100 chance of replying to a message if written by a user in allowedUsers
	randInt := rand.Intn(100)
	log.Println("Random number:", randInt)
	log.Println("Mentions:", m.Mentions)
	if len(m.Mentions) == 0 && randInt == 4 {
		for _, user := range allowedUsers {
			log.Println("User:", user)
			if m.Author.Username == user {
				log.Println("User is in allowedUsers")
				r, err := GenerateGPT4Response(m.Content, m.Author.Username)
				if err != nil {
					log.Println("Failed to get OpenAI response:", err)
					return
				}
				log.Println("OpenAI response:", r)
				log.Println("Channel ID:", m.ChannelID)
				_, err = s.ChannelMessageSend(m.ChannelID, r)
				if err != nil {
					log.Println("Failed to send message to channel:", err)
					return
				}
			}
		}
	}

	if m.Mentions != nil {
		for _, mention := range m.Mentions {
			if mention.ID == s.State.User.ID {
				r, err := GenerateGPT4Response(m.Content, m.Author.Username)
				if err != nil {
					if strings.Contains(err.Error(), "prompt is too long") {
						if _, err := s.ChannelMessageSend(m.ChannelID, "Message is too long!"); err != nil {
							log.Println("Failed to send message to channel:", err)
							return
						}
						return
					}

					if strings.Contains(err.Error(), "prompt is too short") {
						if _, err := s.ChannelMessageSend(m.ChannelID, "Message is too short!"); err != nil {
							log.Println("Failed to send message to channel:", err)
							return
						}
						return
					}

					message, err := s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("Brain broke :flushed: %v", err))
					if err != nil {
						log.Println("Failed to send message to channel:", err)
						return
					}
					_ = message
					return
				}
				_, err = s.ChannelMessageSend(m.ChannelID, r)
				if err != nil {
					log.Println("Failed to send message to channel:", err)
					return
				}
			}
		}
	}

	if strings.HasPrefix(strings.ToLower(m.Content), "lovibot") {
		r, err := GenerateGPT4Response(m.Content, m.Author.Username)
		if err != nil {
			if strings.Contains(err.Error(), "prompt is too long") {
				_, err := s.ChannelMessageSend(m.ChannelID, "Message is too long!")
				if err != nil {
					log.Println("Failed to send message to channel:", err)
					return
				}
				return
			}

			if strings.Contains(err.Error(), "prompt is too short") {
				_, err := s.ChannelMessageSend(m.ChannelID, "Message is too short!")
				if err != nil {
					log.Println("Failed to send message to channel:", err)
					return
				}
				return
			}

			message, err := s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("Brain broke :flushed: %v", err))
			if err != nil {
				log.Println("Failed to send message to channel:", err)
				return
			}
			_ = message
			return
		}
		_, err = s.ChannelMessageSend(m.ChannelID, r)
		if err != nil {
			log.Println("Failed to send message to channel:", err)
			return
		}
	}

}

func main() {
	discordToken := config.DiscordToken

	// Create a new Discord session using the provided bot token.
	session, err := discordgo.New("Bot " + discordToken)
	if err != nil {
		log.Fatalf("Cannot create a new Discord session: %v", err)
	}

	// Add a handler function to the discordgo.Session that is triggered when a slash command is received.
	session.AddHandler(func(s *discordgo.Session, i *discordgo.InteractionCreate) {
		if h, ok := commandHandlers[i.ApplicationCommandData().Name]; ok {
			log.Printf("Handling '%v' command. %+v", i.ApplicationCommandData().Name, i.ApplicationCommandData())
			h(s, i)
		}
	})

	// Add a handler function to the discordgo.Session that is triggered when a message is received.
	session.AddHandler(onMessageCreate)

	// Print the user we are logging in as.
	session.AddHandler(func(s *discordgo.Session, _ *discordgo.Ready) {
		log.Printf("Logged in as: %v#%v", s.State.User.Username, s.State.User.Discriminator)
	})

	// Open a websocket connection to Discord and begin listening.
	err = session.Open()
	if err != nil {
		log.Fatalf("Cannot open the session: %v", err)
	}

	// Remove all existing commands.
	appID := session.State.User.ID
	log.Println("Removing existing commands from all servers for the bot", appID)

	// Remove the commands for guild 98905546077241344
	log.Println("Removing commands for Killyoy's server...")
	commands, err := session.ApplicationCommands(appID, "98905546077241344")
	if err != nil {
		log.Panicf("Cannot get commands for guild 98905546077241344: %v", err)
	}
	for _, v := range commands {
		err := session.ApplicationCommandDelete(session.State.User.ID, "98905546077241344", v.ID)
		if err != nil {
			log.Panicf("Cannot delete '%v' command: %v", v.Name, err)
		}
		log.Printf("Deleted '%v' command.", v.Name)
	}

	// Remove the commands for guild 341001473661992962
	log.Println("Removing commands for TheLovinator's server...")
	commands, err = session.ApplicationCommands(appID, "341001473661992962")
	if err != nil {
		log.Panicf("Cannot get commands for guild 341001473661992962: %v", err)
	}
	for _, v := range commands {
		err := session.ApplicationCommandDelete(session.State.User.ID, "341001473661992962", v.ID)
		if err != nil {
			log.Panicf("Cannot delete '%v' command: %v", v.Name, err)
		}
		log.Printf("Deleted '%v' command.", v.Name)
	}

	// Register the commands for guild 98905546077241344
	log.Println("Registering commands for Killyoy's server...")
	registeredCommands := make([]*discordgo.ApplicationCommand, len(commands))
	for i, v := range commands {
		cmd, err := session.ApplicationCommandCreate(session.State.User.ID, "98905546077241344", v)
		if err != nil {
			log.Panicf("Cannot create '%v' command: %v", v.Name, err)
		}
		registeredCommands[i] = cmd
		log.Printf("Registered '%v' command.", cmd.Name)
	}

	// Register the commands for guild 341001473661992962
	log.Println("Registering commands for TheLovinator's server...")
	registeredCommands = make([]*discordgo.ApplicationCommand, len(commands))
	for i, v := range commands {
		cmd, err := session.ApplicationCommandCreate(session.State.User.ID, "341001473661992962", v)
		if err != nil {
			log.Panicf("Cannot create '%v' command: %v", v.Name, err)
		}
		registeredCommands[i] = cmd
		log.Printf("Registered '%v' command.", cmd.Name)
	}

	// Run s.Close() when the program exits.
	defer session.Close()

	// Wait here until CTRL-C or other term signal is received.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	log.Println("Press Ctrl+C to exit")
	<-stop

	// Bye bye!
	log.Println("Gracefully shutting down.")
}
