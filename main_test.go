package main

import (
	"os"
	"testing"
)

// Returns a string with a length between 1 and 2000 characters
func TestGetPostsFromReddit_ReturnsPostWithValidLength(t *testing.T) {
	if os.Getenv("CI") == "" {
		subreddit := "celebs"
		post, err := GetPostsFromReddit(subreddit)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if len(post) < 1 || len(post) > 2000 {
			t.Errorf("Post length is not within the valid range")
		}
	} else {
		t.Skip("Skipping test in CI environment as the IP is probably blocked by Reddit")
	}
}

// Returns an error when the subreddit does not exist
func TestGetPostsFromReddit_ReturnsErrorWhenSubredditDoesNotExist(t *testing.T) {
	subreddit := "nonexistent"
	_, err := GetPostsFromReddit(subreddit)
	if err == nil {
		t.Errorf("Expected error, but got nil")
	}
}

// Returns an error when the subreddit is empty
func TestGetPostsFromReddit_ReturnsErrorWhenSubredditIsEmpty(t *testing.T) {
	subreddit := ""
	_, err := GetPostsFromReddit(subreddit)
	if err.Error() != "subreddit cannot be empty" {
		t.Errorf("Expected error 'subreddit cannot be empty', but got '%v'", err)
	}
}
