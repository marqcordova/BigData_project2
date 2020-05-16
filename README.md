# BigData_project2
CNN on cancer histopath data

Uploaded a prep file.
That way we don't have to transfer all of the files back and forth, we all get the same train, val and test sets.

Uploaded a training file.
It's the most simple one possible. With a few tweaks (dropout and vert/horiz flipping) we get up to ~93.5 val accuracy.

# Miscellaneous Git Notes from Jack
 - I added a .gitignore file. On your local machine, you can copy the training/testing data into the /data folder and sleep soundly knowing that you'll never accidentally commit >5GB of training images!
 - Here is a list of useful git commands:

    1. `git pull` - Downloads the latest changes from the GitHub (remote) repository.
    2. `git stash` - If git pull complains, you can usually "stash" your changes with this command.
    3. `git stash pop` - "Unstashes" your previously-stashed changes.
    4. `git status` - Displays a list of changed files in your repository.
    5. `git add -A` - Stages all recently-changed files for commit.
    6. `git commit -m "<your message here>"` - Commits any staged changes with a message. Example: `git commit -m "Updated README.md with new contact info."`
    7. `git push` - Uploads your commits to the remote GitHub repository.
    8. `git clone https://github.com/marqcordova/BigData_project2` - Clones this repo onto your machine.

  - A basic git workflow typically looks like this:

    1. Clone the repo onto your machine using `git clone`.
    2. Make changes to the repo with your favorite IDE or code editor. You can edit existing files, add files, remove files, etc.
    3. Commit your changes. Run `git add -A && git commit -m "<add a commit message describing changes here>"`
    4. Push your changes to the remote repository with `git push`.
    5. Go to step 2 and repeat.
