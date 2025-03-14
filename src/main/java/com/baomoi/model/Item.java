package com.baomoi.model;


import org.jetbrains.annotations.NotNull;

public class Item implements Comparable<Item> {
    private String articleLink;
    private String websiteSource;
    private String articleType;
    private String articleTitle;
    private String content;
    private String creationDate;
    private String author;
    private String category;
    private String tags;
    private String summary;

    // Constructors
    public Item() {
        // Default constructor
    }

    public Item(String articleLink, String websiteSource, String articleType, String articleTitle,
                String content, String creationDate, String author, String category, String tags,
                String summary) {
        this.articleLink = articleLink;
        this.websiteSource = websiteSource;
        this.articleType = articleType;
        this.articleTitle = articleTitle;
        this.content = content;
        this.creationDate = creationDate;
        this.author = author;
        this.category = category;
        this.tags = tags;
        this.summary = summary;
    }
    // Getters and setters for all properties
    public String getArticleLink() {
        return articleLink;
    }

    public void setArticleLink(String articleLink) {
        this.articleLink = articleLink;
    }

    public String getWebsiteSource() {
        return websiteSource;
    }

    public void setWebsiteSource(String websiteSource) {
        this.websiteSource = websiteSource;
    }

    public String getArticleType() {
        return articleType;
    }

    public void setArticleType(String articleType) {
        this.articleType = articleType;
    }

    public String getArticleTitle() {
        return articleTitle;
    }

    public void setArticleTitle(String articleTitle) {
        this.articleTitle = articleTitle;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getCreationDate() {
        return creationDate;
    }

    public void setCreationDate(String creationDate) {
        this.creationDate = creationDate;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getTags() {
        return tags;
    }

    public void setTags(String tags) {
        this.tags = tags;
    }

    public String getSummary() {
        return summary;
    }

    public void setSummary(String summary) {
        this.summary = summary;
    }

    @Override
    public int compareTo(@NotNull Item o) {
        return 0;
    }
}
