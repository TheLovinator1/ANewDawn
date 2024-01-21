package main

import (
	"log"
	"os"
	"os/signal"

	"github.com/bwmarrin/discordgo"
	"github.com/joho/godotenv"
)

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

func main() {
	/*
	   Load the .env file into the environment.

	   You can create a .env file with the following contents:
	   TOKEN=your-bot-token-here
	*/
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
	session, err := discordgo.New("Bot " + token)
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