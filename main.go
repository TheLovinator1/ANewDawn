package main

import (
	"log"
	"os"
	"os/signal"

	"github.com/bwmarrin/discordgo"
	"github.com/joho/godotenv"
)

// A Session represents a connection to the Discord API.
var s *discordgo.Session

/*
/ping - The bot responds with "Pong!".
*/
var (
	commands = []*discordgo.ApplicationCommand{
		{
			Name:        "ping",
			Description: "Pong!",
		},
	}

	commandHandlers = map[string]func(s *discordgo.Session, i *discordgo.InteractionCreate){
		// Ping command
		"ping": func(s *discordgo.Session, i *discordgo.InteractionCreate) {
			s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
				Type: discordgo.InteractionResponseChannelMessageWithSource,
				Data: &discordgo.InteractionResponseData{
					Content: "Pong!",
				},
			})
		},
	}
)

/*
init() is automatically called when the package is initialized.
It adds a handler function to the discordgo.Session that is triggered when a slash command is received.
The handler function checks if the interaction's application command name exists in the commandHandlers map.
If it does, it calls the corresponding function from the map with the session and interaction as arguments.
*/
func init() {
	// Load .env file.
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Cannot load .env file: %v", err)
	}

	// Get the bot token from the environment.
	token := os.Getenv("TOKEN")
	if token == "" {
		log.Fatalln("No token provided. Please set the TOKEN environment variable.")
	}

	// Create a new Discord session using the provided bot token.
	s, err = discordgo.New("Bot " + token)
	if err != nil {
		log.Fatalf("Cannot create a new Discord session: %v", err)
	}

	// Add a handler function to the discordgo.Session that is triggered when a slash command is received.
	s.AddHandler(func(s *discordgo.Session, i *discordgo.InteractionCreate) {
		if h, ok := commandHandlers[i.ApplicationCommandData().Name]; ok {
			log.Printf("Handling '%v' command.", i.ApplicationCommandData().Name)
			h(s, i)
		}
	})
}

func main() {
	// Print the user we are logging in as.
	s.AddHandler(func(s *discordgo.Session, _ *discordgo.Ready) {
		log.Printf("Logged in as: %v#%v", s.State.User.Username, s.State.User.Discriminator)
	})

	// Open a websocket connection to Discord and begin listening.
	err := s.Open()
	if err != nil {
		log.Fatalf("Cannot open the session: %v", err)
	}

	// Register the commands.
	log.Println("Adding commands...")
	registeredCommands := make([]*discordgo.ApplicationCommand, len(commands))
	for i, v := range commands {
		cmd, err := s.ApplicationCommandCreate(s.State.User.ID, "341001473661992962", v)
		if err != nil {
			log.Panicf("Cannot create '%v' command: %v", v.Name, err)
		}
		registeredCommands[i] = cmd
		log.Printf("Registered '%v' command.", cmd.Name)
	}

	// Run s.Close() when the program exits.
	defer s.Close()

	// Wait here until CTRL-C or other term signal is received.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	log.Println("Press Ctrl+C to exit")
	<-stop

	// Bye bye!
	log.Println("Gracefully shutting down.")
}
