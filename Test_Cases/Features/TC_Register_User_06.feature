# ***************************************************************************************************
# File        : TC_Register_05.feature
# Description : Feature for registering a new user on ZARA site using AI locator healing.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-07-18
# ***************************************************************************************************

Feature: Register a new user on ZARA

  As a new visitor
  I want to register a new account on the Zara website
  So that I can access a personalised shopping experience

  Background:
    Given the user has launched the ZARA site and reject cookie

  Scenario: Register with valid personal details
    When the user clicks on the "REGISTER" button on the home page
    And the user enters "aniket.pathare@email.com" in the "E-MAIL" field
    And the user enters "Test@1234" in the "PASSWORD" field
    And the user enters "Aniket" in the "NAME" field
    And the user enters "Pathare" in the "SURNAME" field
    And the user enters "+353" as "PREFIX" and "123456789" as "TELEPHONE" field
    And the user accepts the privacy and cookies policy
    And the user clicks on the "CREATE ACCOUNT" button
    Then a confirmation message should be displayed for successful registration
