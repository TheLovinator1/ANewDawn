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
	if err != nil {
		s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
			Type: discordgo.InteractionResponseChannelMessageWithSource,
			Data: &discordgo.InteractionResponseData{
				Content: fmt.Sprintf("Cannot get a random post: %v", err),
				Flags:   discordgo.MessageFlagsEphemeral,
			},
		})
		return
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: post,
		},
	})
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
		{
			Name:        "help",
			Description: "Sends help message",
		},
		{
			Name:        "echo",
			Description: "Echoes your message",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "message",
					Description: "The message to echo",
					Required:    true,
				},
			},
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
		// Help command
		"help": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			// Send the help message to the channel where the command was used
			s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
				Type: discordgo.InteractionResponseChannelMessageWithSource,
				Data: &discordgo.InteractionResponseData{
					Content: "**Commands**\n\n/dank_memes - Sends dank meme from /r/GoodAnimemes\n/waifus - Sends waifu from /r/WatchItForThePlot\n/milkers - Sends milkers from /r/RetrousseTits\n/thighs - Sends thighs from /r/ZettaiRyouiki\n/help - Sends help message\n/echo - Echoes your message",
				},
			})
		},
		// Echo command
		"echo": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			// Check if the user provided a message
			if len(i.ApplicationCommandData().Options) == 0 {
				// If not, send an ephemeral message to the user
				s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
					Type: discordgo.InteractionResponseChannelMessageWithSource,
					Data: &discordgo.InteractionResponseData{
						Content: "You need to provide a message!",
						Flags:   discordgo.MessageFlagsEphemeral,
					},
				})
				return
			}

			// Check that the option contains text
			if i.ApplicationCommandData().Options[0].Type != discordgo.ApplicationCommandOptionString {
				// If not, send an ephemeral message to the user
				s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
					Type: discordgo.InteractionResponseChannelMessageWithSource,
					Data: &discordgo.InteractionResponseData{
						Content: "The message needs to be text!",
						Flags:   discordgo.MessageFlagsEphemeral,
					},
				})
				return
			}

			// Check that the option is not empty
			if i.ApplicationCommandData().Options[0].StringValue() == "" {
				// If not, send an ephemeral message to the user
				s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
					Type: discordgo.InteractionResponseChannelMessageWithSource,
					Data: &discordgo.InteractionResponseData{
						Content: "The message cannot be empty!",
						Flags:   discordgo.MessageFlagsEphemeral,
					},
				})
				return
			}

			// Respond to the original message so we don't get "This interaction failed" error
			s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
				Type: discordgo.InteractionResponseChannelMessageWithSource,
				Data: &discordgo.InteractionResponseData{
					Content: "love u",
					Flags:   discordgo.MessageFlagsEphemeral,
				},
			})

			// Send the message to the channel where the command was used
			s.ChannelMessageSend(i.ChannelID, i.ApplicationCommandData().Options[0].StringValue())
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
				r, err := GetGPT3Response(m.Content, m.Author.Username)
				if err != nil {
					log.Println("Failed to get GPT-3 response:", err)
					return
				}
				log.Println("GPT-3 response:", r)
				log.Println("Channel ID:", m.ChannelID)
				s.ChannelMessageSend(m.ChannelID, r)
			}

		}
	}

	if m.Mentions != nil {
		for _, mention := range m.Mentions {
			if mention.ID == s.State.User.ID {
				r, err := GetGPT3Response(m.Content, m.Author.Username)
				if err != nil {
					if strings.Contains(err.Error(), "prompt is too long") {
						s.ChannelMessageSend(m.ChannelID, "Message is too long!")
						return
					}

					if strings.Contains(err.Error(), "prompt is too short") {
						s.ChannelMessageSend(m.ChannelID, "Message is too short!")
						return
					}

					s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("Brain broke :flushed: %v", err))
					return
				}
				s.ChannelMessageSend(m.ChannelID, r)
			}
		}
	}

	if strings.HasPrefix(strings.ToLower(m.Content), "lovibot") {
		r, err := GetGPT3Response(m.Content, m.Author.Username)
		if err != nil {
			if strings.Contains(err.Error(), "prompt is too long") {
				s.ChannelMessageSend(m.ChannelID, "Message is too long!")
				return
			}

			if strings.Contains(err.Error(), "prompt is too short") {
				s.ChannelMessageSend(m.ChannelID, "Message is too short!")
				return
			}

			s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("Brain broke :flushed: %v", err))
			return
		}
		s.ChannelMessageSend(m.ChannelID, r)
	}

}

func main() {
	// Print the token for debugging purposes.
	discordToken := config.DiscordToken
	fmt.Println("Discord Token:", discordToken)

	// Create a new Discord session using the provided bot token.
	session, err := discordgo.New("Bot " + discordToken)
	if err != nil {
		log.Fatalf("Cannot create a new Discord session: %v", err)
	}

	// Add a handler function to the discordgo.Session that is triggered when a slash command is received.
	session.AddHandler(func(s *discordgo.Session, i *discordgo.InteractionCreate) {
		if h, ok := commandHandlers[i.ApplicationCommandData().Name]; ok {
			log.Printf("Handling '%v' command.", i.ApplicationCommandData().Name)
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

	// Register the commands.
	log.Println("Adding commands...")
	registeredCommands := make([]*discordgo.ApplicationCommand, len(commands))
	for i, v := range commands {
		cmd, err := session.ApplicationCommandCreate(session.State.User.ID, "", v)
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
