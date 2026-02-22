use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "brain")]
#[command(about = "Local RAG brain for code and documentation", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Index {
        #[command(subcommand)]
        source: IndexCommands,
    },
    Update {
        #[command(subcommand)]
        action: UpdateCommands,
    },
    Query {
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: usize,
        #[arg(short, long)]
        json: bool,
    },
    Search {
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    Serve {
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    Sources {
        #[arg(short, long)]
        json: bool,
    },
    Delete {
        source: String,
    },
    Status,
}

#[derive(Subcommand)]
pub enum IndexCommands {
    Github {
        owner: String,
        repo: String,
        #[arg(short, long, default_value = "main")]
        branch: String,
    },
    Local {
        path: PathBuf,
    },
    Defaults,
}

#[derive(Subcommand)]
pub enum UpdateCommands {
    Check {
        #[arg(short, long)]
        json: bool,
    },
    Run {
        #[arg(short, long)]
        force: bool,
        #[arg(short, long)]
        json: bool,
    },
    Status,
}
